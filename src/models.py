import time
import scipy
from _shared_imports import *
from torch import nn
from torch.nn.init import _calculate_correct_fan, calculate_gain

from graph import compute_WL_graph_embeddings, create_gcondnet_graph_dataset, compute_graph_embeddings, create_edges_graphs_sparse_relative_distance, load_random_graph, create_edges_graphs_knn
from graph import GCNEncoder
import torch_geometric as tg
from torch_geometric.nn.models import GAE

####################################  UTILS  ####################################
def tile_weights(args, embeddings):
    """
    This function tiles the embeddings to match the shape of the feature extractor first layer.
    """
    if args.feature_extractor_dims[0] > embeddings.shape[0]:
        embeddings = embeddings.repeat(args.feature_extractor_dims[0]//embeddings.shape[0], 1)
        embeddings = torch.cat([embeddings, embeddings[:args.feature_extractor_dims[0] % embeddings.shape[0], :]], dim=0)
   
    return embeddings

def get_labels_lists(outputs):
    all_y_true, all_y_pred = [], []
    for output in outputs:
        all_y_true.extend(output['y_true'].detach().cpu().numpy().tolist())
        all_y_pred.extend(output['y_pred'].detach().cpu().numpy().tolist())

    return all_y_true, all_y_pred

def detach_tensors(tensors):
    """
    Detach losses 
    """
    if type(tensors)==list:
        detached_tensors = list()
        for tensor in tensors:
            detached_tensors.append(tensor.detach())
        return detached_tensors
    elif type(tensors)==dict:
        detached_tensors = dict()
        for key, tensor in tensors.items():
            detached_tensors[key] = tensor.detach()
        return detached_tensors
    else:
        raise Exception("tensors must be a list or a dict")

def reshape_batch(batch):
    """
    When the dataloaders create multiple samples from one original sample, the input has size (batch_size, no_samples, D)
    
    This function reshapes the input from (batch_size, no_samples, D) to (batch_size * no_samples, D)
    """
    x, y = batch
    x = x.reshape(-1, x.shape[-1])
    y = y.reshape(-1)

    return x, y

def compute_all_metrics(args, y_true, y_pred):
    metrics = {}
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)

    return metrics

# compute the std of the Kaiming initilisation (used to determine the std of the weight initialisations)
def compute_kaiming_normal_std(weights, a=0.01, mode='fan_out', nonlinearity='leaky_relu'):
    fan = _calculate_correct_fan(weights, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    return std

def linear_interpolation_coefficient(max_iterations, iteration):
    if max_iterations==None or max_iterations<=0:
        raise Exception("max_iterations must be a positive integer")

    if iteration > max_iterations:
        return 0
    else:
        return 1 - iteration / max_iterations

""""
General components of a model

WeightPredictorNetwork(optional) -> FeatureExtractor -> Decoder (optional) -> DNN
"""
def create_model(args, data_module=None):
    """
    Function to create the model. Firstly creates the components (e.g., FeatureExtractor, Decoder) and then assambles them.

    Returns a model instance.
    """
    pl.seed_everything(args.seed_model_init_and_training, workers=True)
    
    ### create embedding matrices
    wpn_embedding_matrix = data_module.get_embedding_matrix(args.wpn_embedding_type, args.wpn_embedding_size)

    ### create decoder
    if args.gamma > 0:
        wpn_decoder = WeightPredictorNetwork(args, wpn_embedding_matrix)
        decoder = Decoder(args, wpn_decoder)
    else:
        decoder = None
    
    # compute the standard deviation for the kaiming initialisation of the first layer
    # 	used for some weight initialisations and virtual layers
    std_kaiming_first_layer = compute_kaiming_normal_std(torch.zeros(args.feature_extractor_dims[0], args.num_features))

    ### create models
    if args.model=='fsnet':
        concrete_layer = ConcreteLayer(args, args.num_features, args.feature_extractor_dims[0], is_diet_layer=True, wpn_embedding_matrix=wpn_embedding_matrix)

        return DNN(args, concrete_layer, decoder=decoder)

    elif args.model=='cae': # Supervised Autoencoder
        concrete_layer = ConcreteLayer(args, args.num_features, args.feature_extractor_dims[0])

        return DNN(args, concrete_layer)

    elif args.model in ['dnn', 'dietdnn']:
        if args.model=='dnn':
            layer_type = 'standard'
        
            # create matrix for initialisation
            if args.winit_initialisation=='gcondnet':
                #### CREATE GRAPHS DATASETS
                # SELECT THE TYPE OF EDGES
                match args.winit_graph_connectivity_type:
                    case 'sparse-relative-distance':
                        list_of_edges = create_edges_graphs_sparse_relative_distance(data_module.X_train_raw, ratio_diff_to_neighbor=0.05, max_degree=25)
                    case 'random':
                        list_of_edges = load_random_graph(args.dataset, repeat_id=args.winit_random_graph_repeat_id)
                    case 'knn':
                        list_of_edges = create_edges_graphs_knn(data_module.X_train_raw, k=5)
                    case _:
                        raise Exception("Select a valid graph connectivity type.")

                # CREATE THE NODE FEATURES AND THE GRAPHS THEMSELVES				
                graphs_dataset_all = create_gcondnet_graph_dataset(
                    data_module.X_train, 
                    list_of_edges = list_of_edges
                )

                in_channels = graphs_dataset_all.data[0].x.shape[1]
                out_channels = args.winit_graph_embedding_size

                # data[0].x represents the node feature matrix with shape [num_nodes, num_node_features]
                gnn = GAE(GCNEncoder(in_channels, out_channels, dropout_rate=0.5)).cuda()
                
                data_module.graphs_dataset = graphs_dataset_all

                # put all graphs into one batch for easy forward passing through the GNN
                gnn_dataloader = tg.loader.DataLoader(graphs_dataset_all, batch_size=len(graphs_dataset_all), shuffle=False)
                gnn_batch_train = next(iter(gnn_dataloader))
                
                gnn_batch_train.edge_index = tg.utils.coalesce(gnn_batch_train.edge_index, is_sorted=False)

                first_layer = FirstLinearLayer(args,
                    layer_type='diet_gnn', alpha_interpolation=args.winit_first_layer_interpolation, sparsity_type=args.sparsity_type,
                    # kwargs arguments
                    gnn_model=gnn, gnn_batch = gnn_batch_train, std_kaiming_normal=std_kaiming_first_layer)

                return DNN(args, first_layer, decoder=decoder)

            if args.winit_initialisation=='pca': # PCA-based weight initialisation
                # create the weights
                print("Creating PCA-based weight initialisation....")
                initial_weights = data_module.get_embedding_matrix('svd', embedding_size=args.feature_extractor_dims[0]).T
                initial_weights = tile_weights(args, initial_weights)

                # zero mean
                initial_weights -= initial_weights.mean()
                # make the standard deviation of the weights be a multiple of the Kaiming_normal standard deviation
                initial_weights *= std_kaiming_first_layer / initial_weights.std().item()
                print("Completed SVD-based weight initialisation")

            elif args.winit_initialisation=='nmf':
                print("Creating NMF-based weight initialisation....")
                initial_weights = data_module.get_embedding_matrix('nmf', embedding_size=args.feature_extractor_dims[0]).T
                initial_weights = tile_weights(args, initial_weights)

                # zero mean
                initial_weights -= initial_weights.mean()
                # make the standard deviation of the weights be a multiple of the Kaiming_normal standard deviation
                initial_weights *= std_kaiming_first_layer / initial_weights.std().item()
                print("Completed NMF-based weight initialisation")

            elif args.winit_initialisation=='wl': # Weisfeiler-Lehman-based weight initialisation
                # embeddings of size (20, D)
                print("Creating Weisfeler-Lehman weight initialisation....")
                start_time = time.time()
                initial_weights, _ = compute_WL_graph_embeddings(data_module.X_train_raw, embeddings_size=args.feature_extractor_dims[0])
                initial_weights = tile_weights(args, initial_weights)

                # zero mean
                initial_weights -= initial_weights.mean()
                # make the standard deviation of the weights be a multiple of the Kaiming_normal standard deviation
                initial_weights *= std_kaiming_first_layer / initial_weights.std().item()

                print("Completed Weisfeler-Lehman weight initialisation in {} seconds".format(time.time() - start_time))
            else:
                initial_weights = None

        elif args.model=='dietdnn':
            layer_type = 'diet'
            initial_weights = None

        first_layer = FirstLinearLayer(args, layer_type=layer_type, sparsity_type=args.sparsity_type,
                        wpn_embedding_matrix=wpn_embedding_matrix,
                        initial_weights=initial_weights, alpha_interpolation=args.winit_first_layer_interpolation)

        return DNN(args, first_layer, decoder=decoder)
    else:
        raise Exception(f"The model ${args.model}$ is not supported")


# ----------------------  Predicting the first layer -- Weight Predictor Network  -------------------
class WeightPredictorNetwork(nn.Module):
    """
    Linear -> Tanh -> Linear -> Tanh
    """
    def __init__(self, args, embedding_matrix):
        """
        A tiny network outputs that outputs a matrix W.

        :param nn.Tensor(D, M) embedding_matrix: matrix with the embeddings (D = number of features, M = embedding size)
        """
        super().__init__()
        print(f"Initializing WeightPredictorNetwork with embedding_matrix of size {embedding_matrix.size()}")
        self.args = args

        self.register_buffer('embedding_matrix', embedding_matrix) # store the static embedding_matrix


        ##### Weight predictor network (wpn)
        layers = []
        prev_dimension = args.wpn_embedding_size
        for i, dim in enumerate(args.diet_network_dims):
            if i == len(args.diet_network_dims)-1: # last layer
                layer = nn.Linear(prev_dimension, dim)
                nn.init.uniform_(layer.weight, -0.01, 0.01) # same initialization as in the DietNetwork original paper
                layers.append(layer)
                layers.append(nn.Tanh())
            else:
                if args.nonlinearity_weight_predictor=='tanh':
                    layer = nn.Linear(prev_dimension, dim)
                    nn.init.uniform_(layer.weight, -0.01, 0.01) # same initialization as in from the DietNetwork original paper
                    layers.append(layer)
                    layers.append(nn.Tanh())					# DietNetwork paper uses tanh all over
                elif args.nonlinearity_weight_predictor=='leakyrelu':
                    layer = nn.Linear(prev_dimension, dim)
                    nn.init.kaiming_normal_(layer.weight, a=0.01, mode='fan_out', nonlinearity='leaky_relu')
                    layers.append(layer)
                    layers.append(nn.LeakyReLU())

                if args.batchnorm:
                    layers.append(nn.BatchNorm1d(dim))
                layers.append(nn.Dropout(args.dropout_rate))
                
            prev_dimension = dim

        self.wpn = nn.Sequential(*layers)
    
    def forward(self):
        # use the wpn to predict the weight matrix
        embeddings = self.embedding_matrix
    
        W = self.wpn(embeddings) # W has size (D x K)
        
        if self.args.softmax_diet_network:
            W = F.softmax(W, dim=1) # FsNet applied softmax over the (feature - all-K-neurons)
        
        return W.T # size K x D


class ZeroWeightPredictorNetwork(nn.Module):
    """
    Outputs a weight matrix W with all zeros
    """
    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self):
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        return torch.zeros(self.args.feature_extractor_dims[0], self.args.num_features, device=device)


class GNNWeightPredictorNetwork(nn.Module):
    """
    Outputs a weight matrix W using a GNN
    """
    def __init__(self, args, model, gnn_batch, std_kaiming_normal):
        super().__init__()
        self.args = args
        self.model = model
        self.gnn_batch = gnn_batch
        self.std_kaiming_normal = std_kaiming_normal

    def forward(self):
        # return a weight matrix
        graph_embeddings = compute_graph_embeddings(self.model, self.gnn_batch) # (K, D)

        # standardise the weights to zero mean and standard deviation multiple of the Kaiming variance       
        weights = (graph_embeddings - graph_embeddings.mean(dim=0)) / (graph_embeddings.std(dim=0) + 1e-6) # add a small epsilon to avoid division by zero
        weights *= self.std_kaiming_normal

        return weights


class FirstLinearLayer(nn.Module):
    """
    First linear layer (with activation, batchnorm and dropout), with the ability to include:
    - diet layer (i.e., there's a weight predictor network which predicts the weight matrix)
    - sparsity network (i.e., there's a sparsity network which outputs sparsity weights)
    """

    def __init__(self, args, layer_type, sparsity_type, initial_weights=None, alpha_interpolation=None, **kwargs):
        """
        If is_diet_layer==None and sparsity_type==None, this layers acts as a standard linear layer

        :param layer_type: type of the layer (e.g., "standard", "diet", "diet_gnn")
        :param sparsity_type: type of the sparsity network ("global") used in the WPFS paper
        :param initial_weights (torch.tensor with gradients enabled): initial weights for the layer (if a standard linear layer is used)
            if None, then use kaiming initialization
        :param alpha_interpolation: interpolation factor for the weight matrix. Used only if layer_type=="diet_gnn"
                    W = alpha W_GNN + (1 - alpha) W_MLP
                    - alpha=0 -> full MLP
                    - alpha=1 -> full GC-MLP end-to-end

        # **kwargs: wpn_embedding_matrix
        """
        super().__init__()

        self.args = args
        self.layer_type = layer_type
        self.sparsity_type = sparsity_type
        self.alpha_interpolation = alpha_interpolation # initial value passed through command line

        # DIET LAYER
        if layer_type=="diet":
            # if diet layer, then initialize a weight predictor network
            
            if self.args.wpn_embedding_type == 'zero': # always return zero
                print("Creating WPN that always returns zero")
                self.wpn = ZeroWeightPredictorNetwork(args)
            else: # learnable WPN that maps an embedding matrix to a weight matrix
                self.wpn = WeightPredictorNetwork(args, kwargs['wpn_embedding_matrix'])

            # used only if alpha_interpolation is not None
            self.weights = nn.Parameter(torch.zeros(args.feature_extractor_dims[0], args.num_features), requires_grad=True)
        elif layer_type in ["diet_gnn"]:
            assert alpha_interpolation and 0 < alpha_interpolation <= 1

            if layer_type == "diet_gnn":
                self.wpn = GNNWeightPredictorNetwork(args, kwargs['gnn_model'], kwargs['gnn_batch'], kwargs['std_kaiming_normal'])
            
            # if the final weight matrix is interpolated between W_GNN and W_scratch
            self.weights = nn.Parameter(torch.zeros(args.feature_extractor_dims[0], args.num_features), requires_grad=True)
        elif layer_type=="standard":
            if initial_weights is None: # initialize weights with kaiming initialization
                self.weights = nn.Parameter(torch.zeros(args.feature_extractor_dims[0], args.num_features), requires_grad=True)
                nn.init.kaiming_normal_(self.weights, a=0.01, mode='fan_out', nonlinearity='leaky_relu')
            else:                       # initialize weights with specified initial weights
                self.weights = nn.Parameter(initial_weights, requires_grad=True)
        else:
            raise ValueError("Invalid first layer type")


        # auxiliary layer after the matrix multiplication
        self.bias_first_layer = nn.Parameter(torch.zeros(args.feature_extractor_dims[0]))
        self.layers_after_matrix_multiplication = nn.Sequential(*[
            nn.LeakyReLU(),
            nn.BatchNorm1d(args.feature_extractor_dims[0]),
            nn.Dropout(args.dropout_rate)
        ])

        # SPARSITY REGULARIZATION for the first layer
        if sparsity_type=='global':
            if args.sparsity_method=='sparsity_network':
                print("Creating Sparsity network")
                self.sparsity_model = SparsityNetwork(args, kwargs['wpn_embedding_matrix'])
            else:
                raise Exception("Sparsity method not valid")
        else:
            self.sparsity_model = None

    def forward(self, x, iteration=None):
        """
        Input:
            x: (batch_size x num_features)
        """

        # COMPUTE WEIGHTS FIRST LAYER
        if self.layer_type in ["diet", 'diet_gnn']:
            alpha_interpolation = self.alpha_interpolation
            if self.args.winit_first_layer_interpolation_scheduler=='linear':
                alpha_interpolation *= linear_interpolation_coefficient(
                    max_iterations = self.args.winit_first_layer_interpolation_end_iteration,
                    iteration = iteration
                )

            if alpha_interpolation == 0: # no WPN, only learned weights
                W = self.weights
            else: # interpolation between WPN-based weight and learned weights
                W_wpn = self.wpn()

                if W_wpn.shape != self.weights.shape:
                    print("W_wpn.shape", W_wpn.shape)
                    print("self.weights.shape", self.weights.shape)
                    raise Exception("W_wpn.shape != self.weights.shape")
                W = alpha_interpolation * W_wpn  + (1 - alpha_interpolation) * self.weights

        elif self.layer_type=="standard":
            W = self.weights # W has size (K x D)

        # APPLY SPARSITY WEIGHTS (from WPFS paper)
        if self.args.sparsity_type==None:
            all_sparsity_weights = None

            hidden_rep = F.linear(x, W, self.bias_first_layer)
        
        elif self.args.sparsity_type=='global':
            all_sparsity_weights = self.sparsity_model(None) 	# Tensor (D, )
            # print("all_sparsity_weights", all_sparsity_weights.shape)
            # print("self.args.num_features", self.args.num_features)
            # print("all_sparsity_weights.shape", all_sparsity_weights.shape)

            assert all_sparsity_weights.shape[0]==self.args.num_features and len(all_sparsity_weights.shape)==1
            W = torch.matmul(W, torch.diag(all_sparsity_weights))

            hidden_rep = F.linear(x, W, self.bias_first_layer)
  
        RESULT = self.layers_after_matrix_multiplication(hidden_rep)

        return RESULT, all_sparsity_weights


# For WPFS baseline
class SparsityNetwork(nn.Module):
	"""
	Sparsity network

	Architecture
	- same 4 hidden layers of 100 neurons as the DietNetwork (for simplicity)
	- output layer: 1 neuron, sigmoid activation function
	- note: the gating network in LSNN used 3 hidden layers of 100 neurons

	Input
	- gene_embedding: gene embedding (batch_size, embedding_size)
	Output
	- sigmoid value (which will get multiplied by the weights associated with the gene)
	"""
	def __init__(self, args, embedding_matrix):
		"""
		:param nn.Tensor(D, M) embedding_matrix: matrix with the embeddings (D = number of features, M = embedding size)
		"""
		super().__init__()
		
		print(f"Initializing SparsityNetwork with embedding_matrix of size {embedding_matrix.size()}")
		
		self.args = args
		self.register_buffer('embedding_matrix', embedding_matrix) # store the static embedding_matrix

		layers = []
		dim_prev = args.wpn

		for _, dim in enumerate(args.diet_network_dims):
			layers.append(nn.Linear(dim_prev, dim))
			layers.append(nn.LeakyReLU())
			layers.append(nn.BatchNorm1d(dim))
			layers.append(nn.Dropout(args.dropout_rate))

			dim_prev = dim
		
		layers.append(nn.Linear(dim, 1))
		self.network = nn.Sequential(*layers)

		if args.mixing_layer_size:
			mixing_layers = []

			layer1 = nn.Linear(args.num_features, args.mixing_layer_size, bias=False)
			nn.init.uniform_(layer1.weight, -0.005, 0.005)
			mixing_layers.append(layer1)

			mixing_layers.append(nn.LeakyReLU())

			if args.mixing_layer_dropout:
				mixing_layers.append(nn.Dropout(args.mixing_layer_dropout))
			
			layer2 = nn.Linear(args.mixing_layer_size, args.num_features, bias=False)
			nn.init.uniform_(layer2.weight, -0.005, 0.005)
			mixing_layers.append(layer2)

			self.mixing_layers = nn.Sequential(*mixing_layers)
		else:
			self.mixing_layers = None

	def forward(self, input):
		"""
		Input:
		- input: Tensor of patients (B, D)

		Returns:
		if args.sparsity_type == 'global':
			- Tensor of sigmoid values (D)
		"""
		if self.args.sparsity_type == 'global':
			out = self.network(self.embedding_matrix) # (D, 1)]

			print("SparsityNetwork: global sparsity")
			print("self.embedding_matrix", self.embedding_matrix.shape)
			print("out", out.shape)

			if self.mixing_layers:
				out = self.mixing_layers(out.T).T # input of size (1, D) to the linear layer

			out = torch.sigmoid(out)
			return torch.squeeze(out, dim=1) 		  # (D)


# ----------------------  Miscellaneous layers/functions  -------------------
class ConcreteLayer(nn.Module):
    """
    Implementation of a concrete layer from paper "Concrete Autoencoders for Differentiable Feature Selection and Reconstruction"
    """

    def __init__(self, args, input_dim, output_dim, is_diet_layer=False, wpn_embedding_matrix=None):
        """
        - input_dim (int): dimension of the input
        - output_dim (int): number of neurons in the layer
        """
        super().__init__()
        self.args = args
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.temp_start = 10
        self.temp_end = 0.01
        # the iteration is used in annealing the temperature
        # 	it's increased with every call to sample during training
        self.current_iteration = 0 
        self.anneal_iterations = args.concrete_anneal_iterations # maximum number of iterations for the temperature optimization

        self.is_diet_layer = is_diet_layer
        if is_diet_layer:
            # if diet layer, then initialize a weight predictor matrix
            assert wpn_embedding_matrix is not None
            self.wpn = WeightPredictorNetwork(args, wpn_embedding_matrix)
        else:
            self.alphas = nn.Parameter(torch.zeros(output_dim, input_dim), requires_grad=True)
            torch.nn.init.xavier_normal_(self.alphas, gain=1) # Glorot normalization, following the original CAE implementation
        
    def get_temperature(self):
        # compute temperature		
        if self.current_iteration >= self.anneal_iterations:
            return self.temp_end
        else:
            return self.temp_start * (self.temp_end / self.temp_start) ** (self.current_iteration / self.anneal_iterations)

    def sample(self):
        """
        Sample from the concrete distribution.
        """
        # Increase the iteration counter during training
        if self.training:
            self.current_iteration += 1

        temperature = self.get_temperature()

        alphas = self.wpn() if self.is_diet_layer else self.alphas # alphas is a K x D matrix

        # sample from the concrete distribution
        if self.training:
            samples = F.gumbel_softmax(alphas, tau=temperature, hard=False) # size K x D
            assert samples.shape == (self.output_dim, self.input_dim)
        else: 			# sample using argmax
            index_max_alphas = torch.argmax(alphas, dim=1) # size K
            samples = torch.zeros(self.output_dim, self.input_dim).cuda()
            samples[torch.arange(self.output_dim), index_max_alphas] = 1.

        return samples

    def forward(self, x):
        """
        - x (batch_size x input_dim)
        """
        mask = self.sample()   	# size (number_neurons x input_dim)
        x = torch.matmul(x, mask.T) 		# size (batch_size, number_neurons)
        return x, None # return additional None for compatibility


def create_linear_layers(args, layer_sizes, layers_for_hidden_representation):
    """
    Args
    - layer_sizes: list of the sizes of the sizes of the linear layers
    - layers_for_hidden_representation: number of layers of the first part of the encoder (used to output the input for the decoder)

    Returns
    Two lists of Pytorch Modules (e.g., Linear, BatchNorm1d, Dropout)
    - encoder_first_part
    - encoder_second_part
    """
    encoder_first_part = []
    encoder_second_part = []
    for i, (dim_prev, dim) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        if i < layers_for_hidden_representation:					# first part of the encoder
            encoder_first_part.append(nn.Linear(dim_prev, dim))
            encoder_first_part.append(nn.LeakyReLU())
            if args.batchnorm:
                encoder_first_part.append(nn.BatchNorm1d(dim))
            encoder_first_part.append(nn.Dropout(args.dropout_rate))
        else:														# second part of the encoder
            encoder_second_part.append(nn.Linear(dim_prev, dim))
            encoder_second_part.append(nn.LeakyReLU())
            if args.batchnorm:
                encoder_second_part.append(nn.BatchNorm1d(dim))
            encoder_second_part.append(nn.Dropout(args.dropout_rate))
        
    return encoder_first_part, encoder_second_part


class Decoder(nn.Module):
    def __init__(self, args, wpn):
        super().__init__()
        assert wpn!=None, "The decoder is used only with a WPN (because it's only used within the DietNetwork)"

        self.wpn = wpn
        self.bias = nn.Parameter(torch.zeros(args.num_features,))

    def forward(self, hidden_rep):
        W = self.wpn().T # W has size D x K

        return F.linear(hidden_rep, W, self.bias)


# ----------------------  DNN class for training  -------------------
class TrainingLightningModule(pl.LightningModule):
    """
    General class to be inherited by all implemented models (e.g., MLP, CAE, FsNet etc.)

    It implements general training and evaluation functions (e.g., computing losses, logging, training etc.)
    """
    def __init__(self, args):
        super().__init__()
        self.args = args

    def compute_loss(self, y_true, y_hat, x, x_hat, sparsity_weights):
        losses = {}
        losses['cross_entropy'] = F.cross_entropy(input=y_hat, target=y_true, weight=torch.tensor(self.args.class_weights, device=self.device))
        losses['reconstruction'] = F.mse_loss(x_hat, x, reduction='mean') if self.decoder else torch.zeros(1, device=self.device)

        ### sparsity loss
        if sparsity_weights is None:
            losses['sparsity'] = torch.tensor(0., device=self.device)
        else:
            if self.args.sparsity_regularizer=='L1':
                losses['sparsity'] = self.args.sparsity_regularizer_hyperparam * torch.norm(sparsity_weights, 1)
            elif self.args.sparsity_regularizer=='hoyer':
                hoyer_reg = torch.norm(sparsity_weights, 1) / torch.norm(sparsity_weights, 2)
                losses['sparsity'] = self.args.sparsity_regularizer_hyperparam * hoyer_reg
            else:
                raise Exception("Sparsity regularizer not valid")

        losses['total'] = losses['cross_entropy'] + self.args.gamma * losses['reconstruction'] + losses['sparsity']
        
        return losses

    def log_losses(self, losses, key, dataloader_name=""):
        self.log(f"{key}/total_loss{dataloader_name}", losses['total'].item())
        self.log(f"{key}/reconstruction_loss{dataloader_name}", losses['reconstruction'].item())
        self.log(f"{key}/cross_entropy_loss{dataloader_name}", losses['cross_entropy'].item())
        self.log(f"{key}/sparsity_loss{dataloader_name}", losses['sparsity'].item())
        if 'graph_reconstruction' in losses:
            self.log(f'{key}/graph_reconstruction', losses['graph_reconstruction'].item())


    def log_epoch_metrics(self, outputs, key, dataloader_name=""):
        y_true, y_pred = get_labels_lists(outputs)
        self.log(f'{key}/balanced_accuracy{dataloader_name}', balanced_accuracy_score(y_true, y_pred))

    def training_step(self, batch, batch_idx):
        """
        :param batch: dictionary of batches coming from multiple dataloaders
        """
        # tabular data
        x, y_true = batch['tabular']
        y_hat, x_hat, sparsity_weights = self.forward(x)

        losses = self.compute_loss(y_true, y_hat, x, x_hat, sparsity_weights)

        self.log_losses(losses, key='train')
        
        if isinstance(self.first_layer, ConcreteLayer):
            self.log("train/concrete_temperature", self.first_layer.get_temperature())

        return {
            'loss': losses['total'],
            'losses': detach_tensors(losses),
            'y_true': y_true,
            'y_pred': torch.argmax(y_hat, dim=1)
        }

    def training_epoch_end(self, outputs):
        self.log_epoch_metrics(outputs, 'train')

    def validation_step(self, batch, batch_idx):
        x, y_true = batch['tabular']
        y_hat, x_hat, sparsity_weights = self.forward(x)

        losses = self.compute_loss(y_true, y_hat, x, x_hat, sparsity_weights)

        return {
            'losses': detach_tensors(losses),
            'y_true': y_true,
            'y_pred': torch.argmax(y_hat, dim=1)
        }

    def validation_epoch_end(self, outputs):
        losses = {
            'total': np.mean([output['losses']['total'].item() for output in outputs]),
            'reconstruction': np.mean([output['losses']['reconstruction'].item() for output in outputs]),
            'cross_entropy': np.mean([output['losses']['cross_entropy'].item() for output in outputs]),
            'sparsity': np.mean([output['losses']['sparsity'].item() for output in outputs])
        }

        # sometimes the graph reconstruction loss is not computed
        if 'graph_reconstruction' in outputs[0]['losses']:
            losses['graph_reconstruction'] = np.mean([output['losses']['graph_reconstruction'].item() for output in outputs]),

        self.log_losses(losses, key='valid')
        self.log_epoch_metrics(outputs, key='valid')

    def test_step(self, batch, batch_idx):
        x, y_true = batch['tabular']
        y_hat, x_hat, sparsity_weights = self.forward(x)
        losses = self.compute_loss(y_true, y_hat, x, x_hat, sparsity_weights)

        return {
            'losses': detach_tensors(losses),
            'y_true': y_true,
            'y_pred': torch.argmax(y_hat, dim=1),
            'y_hat': y_hat.detach().cpu().numpy()
        }

    def test_epoch_end(self, outputs):
        ### Save losses
        losses = {
            'total': np.mean([output['losses']['total'].item() for output in outputs]),
            'reconstruction': np.mean([output['losses']['reconstruction'].item() for output in outputs]),
            'cross_entropy': np.mean([output['losses']['cross_entropy'].item() for output in outputs]),	
            'sparsity': np.mean([output['losses']['sparsity'].item() for output in outputs])
        }

        # sometimes the graph reconstruction loss is not computed
        if 'graph_reconstruction' in outputs[0]['losses']:
            losses['graph_reconstruction'] = np.mean([output['losses']['graph_reconstruction'].item() for output in outputs]),

        self.log_losses(losses, key=self.log_test_key)
        self.log_epoch_metrics(outputs, self.log_test_key)

        #### Save prediction probabilities
        y_hat_list = [output['y_hat'] for output in outputs]
        y_hat_all = np.concatenate(y_hat_list, axis=0)
        y_hat_all = scipy.special.softmax(y_hat_all, axis=1)


    def configure_optimizers(self):
        params = self.parameters()

        if self.args.optimizer=='adam':
            optimizer = torch.optim.Adam(params, lr=self.learning_rate, weight_decay=self.args.weight_decay)
        if self.args.optimizer=='adamw':
            optimizer = torch.optim.AdamW(params, lr=self.learning_rate, weight_decay=self.args.weight_decay, betas=[0.9, 0.98])

        if self.args.lr_scheduler == 'none':
            return optimizer
        else:
            raise Exception()


class DNN(TrainingLightningModule):
    """
    Flexible MLP-based architecture which can implement an MLP, WPS, FsNet


    DietDNN architecture
    Linear -> LeakyRelu -> BatchNorm -> Dropout -> Linear -> LeakyRelu-> BatchNorm -> Dropout -> Linear -> y_hat
                                                                                            |
                                                                                            |
                                                                                          hidden
                                                                                            |
                                                                                            v
                                                                                          Linear
                                                                                            |
                                                                                            V
                                                                                          x_hat
    """
    def __init__(self, args, first_layer, second_layer=None, decoder=None):
        """
        DNN with a feature_extractor and a final layer (with `num_classes` logits)
        :param first_layer (nn.Module): first layer of the DNN (used mainly for WPN)
        :param second_layer (nn.Module): second layer of the DNN (enables having a WPN on the second layer)
        :param nn.Module decoder: decoder (for reconstruction loss)
                If None, then don't have a reconstruction loss
        """
        super().__init__(args)

        if decoder:
            print(f'Creating {args.model} with decoder...')
        else:
            print(f'Creating {args.model} without decoder...')

        self.args = args
        self.log_test_key = None
        self.learning_rate = args.lr
        
        self.first_layer = first_layer
        self.second_layer = second_layer

        # split the layers into two parts to 
        encoder_first_layers, encoder_second_layers = create_linear_layers(
            args, args.feature_extractor_dims, args.layers_for_hidden_representation-1) # the -1 in (args.layers_for_hidden_representation - 1) is because we don't consider the first layer

        self.encoder_to_hidden = nn.Sequential(*encoder_first_layers)
        self.hidden_to_logits = nn.Sequential(*encoder_second_layers)

        self.classification_layer = nn.Linear(args.feature_extractor_dims[-1], args.num_classes)
        self.decoder = decoder

    def forward(self, x):
        x, sparsity_weights = self.first_layer(x, self.global_step)			   # pass through the first layer
        if self.second_layer:								   # pass through the second layer
            x = self.second_layer(x) 

        x = self.encoder_to_hidden(x)					       # obtain the hidden representation (defined as a layer in the network)
        x_hat = self.decoder(x) if self.decoder else None      # reconstruction
        x = self.hidden_to_logits(x)

        y_hat = self.classification_layer(x)           		   # classification, returns logits

        return y_hat, x_hat, sparsity_weights