function [test_in, test_out] = GenerateData(n)

    test_in = zeros(n,2);
    test_out = zeros(n,2);
    
    for k = 1:n
        [s_in, s_out] = RandomSample();
        test_in(k,:) = s_in;
        test_out(k,:) = s_out;
    end