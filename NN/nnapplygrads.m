function nn = nnapplygrads(nn)
%NNAPPLYGRADS updates weights and biases with calculated gradients
% nn = nnapplygrads(nn) returns an neural network structure with updated
% weights and biases
    
    for i = 1 : (nn.n - 1)
        dW = nn.learningRate * nn.dW{i};
            
        if (nn.dropConnectFraction > 0 &&(i ~= (nn.n-1)))
            dW = dW.*nn.dropConnectMask{i};
        end
            
        if (nn.momentum > 0)
            nn.vW{i} = nn.momentum*nn.vW{i} + dW;
            dW = nn.vW{i};
        end
        
        nn.W{i} = nn.W{i} - dW;
    end
end