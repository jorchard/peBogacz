function [w,b] = my_w_init(params)
%w,b are cell arrays. Their i^th cell elements correspond the the i^th
%weights/biases of the network. For example the first cell element
%corresponds to the first weights/biases that convey information form input
%to first hidden layer of network.
%params is a structure of parameters
%type is the type of non-linearity
%n_layers is number of layers in network
%neurons is a vector with i^th element being the number of neurons in i^th
%layer

type = params.type;
n_layers = params.n_layers;
w = cell(n_layers,1);
b = cell(n_layers,1);
neurons=params.neurons;

w{1} = [[ 0.63827171,  1.94878605],
        [-1.54809306, -1.08798682],
        [ 0.41531103,  0.40218358],
        [ 1.9545372 ,  0.84283361]];
w{2} = [[-0.142614  , -0.50655311,  0.44099561,  1.17943999],
        [ 1.43606398, -0.29175979,  0.3383928 , -0.4229459 ]];
    
b{1} = [-0.04621402, -0.40947562, -0.06824616,  0.54733387]';
b{2} = [-0.58947249, -0.48776308]';
%     w{i} = unifrnd(-1,1,neurons(i+1),neurons(i)) * norm_w ;
%     b{i} = zeros(neurons(i+1),1) + norm_b * ones(neurons(i+1),1) ;  
end





