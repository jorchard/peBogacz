function [rmse, x_output] = test(in,out,w,b,params)
%function [rmse, error_percent, nrmse] = test(in,out,w,b,params)
%w,b - these are the weights and biases
%params - a structure containing parameters
%in - input data, in an array of size (input data dimension x number of data samples) 
%out - output data, in an array of size (output data dimension x number of data samples)
%rmse - rmse error
%error_percent - percent error (if one-hot catergorical output)
%nrmse - normalised rmse error

iterations = size(in,2);
type = params.type;
n_layers = params.n_layers;
x_output = zeros(size(out));
x = cell(n_layers,1);

for its = 1:iterations
    %make prediction
    x{1} = in(:,its);
    for ii = 2:n_layers
        x{ii} = w{ii-1} * (f(x{ii-1}, type{ii-1})) + b{ii-1} ;
    end     
    x_output(:,its) = x{n_layers};
end
%calculate errors
[rmse] = rms_error(out, x_output);
disp(x_output);
end







