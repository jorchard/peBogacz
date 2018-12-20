function [x,e,its] = generate(out,w,b,params)
%function [x,e,its] = infer_pc(x,w,b,params)
% w,b - these are the weights and biases
% x - Variable nodes: First cell is input layer. Last cell is output layer
% e - Error nodes: First cell empty. Last cell is output layer
% params - a structure containing parameters
it_max = params.it_max;
n_layers = params.n_layers;
type = params.type;
beta = params.beta;
e = cell(n_layers,1);
f_n = cell(n_layers,1);
f_p = cell(n_layers,1);
var = params.var;

%e_history = zeros(it_max, length(b{1}));

x{n_layers} = out;
x{1} = zeros(size(w{1},2),1);

%make a prediciton 
for ii = 2:n_layers-1
    x{ii} = w{ii-1} * ( f( x{ii-1} , type{ii-1}) ) +  b{ii-1} ;
end

%calculate initial errors
for ii=2:n_layers
    [f_n{ii-1},f_p{ii-1}] = f_b( x{ii-1}, type{ii-1}) ;
    e{ii} = (x{ii} - w{ii-1} * ( f_n{ii-1} ) - b{ii-1})/var(ii) ;
end

for i = 1:it_max
    %update varaible nodes
    for ii=2:n_layers-1
        g = ( w{ii}' *  e{ii+1} ) .* f_p{ii} ;
        x{ii} = x{ii} + beta * ( - e{ii} + g );
    end
    x{1} = x{1} + beta * ( w{1}' * e{2} ) .* f_p{1};
    
    %calculate errors
    for ii=2:n_layers
        [f_n{ii-1},f_p{ii-1}] = f_b( x{ii-1}, type{ii-1}) ;
        e{ii} = (x{ii} - w{ii-1} * ( f_n{ii-1} ) - b{ii-1})/var(ii) ;
    end
    %e_history(i,:) = e{2}';
end
its=i;
end





