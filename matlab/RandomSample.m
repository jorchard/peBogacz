function [y, x] = RandomSample()

    
    theta = ( rand()*2 - 1 ) * pi/8;
    rho = rand()*0.8 + 1.5;
    y = [rho-1.8 theta];
    x0 = rho*cos(theta) - 1.7;
    x1 = rho*sin(theta)*0.5;
    x = [x0 x1];