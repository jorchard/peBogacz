% example code

%Contrary to our paper, we run the network in the 'opposite direction'
%here as it makes the code clearer to follow.
%so the input layer here is layer 1, and the output layer is layer l_max
%i.e the first set of weights W_1, would take the input layer values and propagate
%it to layer 2

%weights, biases, variables and errors are all stored in cell arrays.

params.type = 'logsig'; %activation type other  'logsig', 'lin', 'reclin'
params.l_rate =  0.2; % learning rate
params.it_max = 100; % maximum iterations of inference
params.epochs = 100; %500; % number of epochs
params.d_rate = 0; % weight decay parameter
params.beta = 0.2; % euler integration constant

%training data XOR problem
sin = [0 0 1 1; ...
       0 1 0 1];
sout = [1 0 0 1];
%[sin, sout] = GenerateData(500);
%sin = [-0.08027386, -0.30823911];
%sout = [-0.06132582, -0.2608663];
% sin = sin';
% sout = sout';

params.neurons = [2 5 1]; %neurons in each layer
%params.type = {'logsig' 'logsig'}; % 'lin'};
params.type = {'tanh' 'tanh'}; % 'lin'};

params.n_layers = length(params.neurons); % number of layers
var = ones(1, params.n_layers); % puts variance on all layers as 1
var(end)=1; % variance on last layer
params.var=var;

% plot(sin(1,:), sin(2,:), 'r.');
% hold on;
% plot(sout(1,:), sout(2,:), 'b.');
% hold off;

[w_pc, b_pc] = my_w_init2(params);
%% 

plotevery = 50;
run_num=1;

for run = 1:run_num;
    [w_pc, b_pc] = my_w_init2(params); % get weights and biases parameters
    w_ann=w_pc;
    b_ann=b_pc;
    counter =1;
    [rms_error_pc(run,counter)] = test(sin,sout,w_pc,b_pc,params); %test pc
    %[rms_error_ann(run,counter)] = test(sin,sout,w_ann,b_ann,params); %test ann 
    
    %learn
    for epoch = 1:params.epochs
        params.epoch_num = epoch;
        [w_pc,b_pc] = learn_pc(sin,sout,w_pc,b_pc,params); %train pc
        %[w_ann,b_ann] = learn_ann(sin,sout,w_ann,b_ann,params); %train ann
        
        if (epoch/params.epochs)*(params.epochs/plotevery) == ceil((epoch/params.epochs)*(params.epochs/plotevery));
            disp(['run=',num2str(run),' it=',num2str(epoch)]);
            counter = counter+1;
            [rms_error_pc(run,counter)] = test(sin,sout,w_pc,b_pc,params); %test pc
            %[rms_error_ann(run,counter)] = test(sin,sout,w_ann,b_ann,params); %test ann 
        end
    end
end

%% Test
[rms_error_pc, x_gen] = test(sin,sout,w_pc,b_pc,params);



%%
[rms, xout] = test(sin, sout, w_pc, b_pc, params);
plot(sout(1,:), sout(2,:), 'b.');
hold on;
plot(xout(1,:), xout(2,:), 'r.');
iterations = size(sin,2);
for its = 1:iterations
    x = cell(params.n_layers,1);

    %organise data into cells arrays
    x{1} = sin(:,its);
    x_out = sout(:,its);
    %make a prediciton 
    for ii = 2:params.n_layers
        x{ii} = w_pc{ii-1} * ( f( x{ii-1} , params.type{ii-1}) ) +  b_pc{ii-1} ;
    end
    %infer
    x{params.n_layers}  = x_out;
    [x,e,~] = infer_pc(x,w_pc,b_pc,params);
    y = x{end};
    plot(y(1), y(2), 'bo');
    
   %plot([sout(1,k) xout(1,k)], [sout(2,k), xout(2,k)], 'g'); 
   %plot([sout(1,k) xout(1,k)], [sout(2,k), xout(2,k)], 'g'); 
end
hold off;



% leg={'run1','run2','run3','run4'};
% 
% figure('color',[1 1 1]);
% subplot(1,2,1);
% plot(0:50:params.epochs,rms_error_pc')
% xlabel('Iterations')
% ylabel('RMSE')
% title('Predictive coding')
% legend(leg)
% set(gca,'xlim',[0 params.epochs]);
% subplot(1,2,2);
% plot(0:50:params.epochs,rms_error_ann')
% xlabel('Iterations')
% ylabel('RMSE')
% title('Artificial NN')
% legend(leg)
% set(gca,'xlim',[0 params.epochs]);






