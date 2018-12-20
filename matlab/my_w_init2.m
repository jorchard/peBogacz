function [w,b] = my_w_init2(params)
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

w{1} = [    0.3261   -0.4540;
   -0.3906   -0.5110;
    0.3181    0.3108;
    0.3613    0.6377;
   -0.7999   -0.2880];

w{2} = [0.5610    0.3507   -0.9866    0.2043   -0.2265];

    
b{1} = zeros(5,1);
b{2} = [0];
%     w{i} = unifrnd(-1,1,neurons(i+1),neurons(i)) * norm_w ;
%     b{i} = zeros(neurons(i+1),1) + norm_b * ones(neurons(i+1),1) ;  
end





