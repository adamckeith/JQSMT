function bounds = measureOperator(O, inferred_rho, povms)
% Measure operator for an inferred density matrix with constraints
% 
% A tuple of expectations are returned corresponding
% to the minimum and maximum expectation over possible density matrices
% consistent with the povms. 
% A semidefinite program calculates these bounds

%addpath('yalmip')
%addpath(genpath('yalmip'))
%addpath(genpath('C:\Users\ACKWinDesk\Desktop\yalmip'))

dim = size(povms, 1);        % dimension of density matrix
num_povms = size(povms, 3);  % number of povm elements (over all povms)

rho = sdpvar(dim, dim, 'hermitian', 'complex');
bounds = zeros(1,2);

%%% Constraints for both bounds %%%
C = [];
C = [rho >= 0, trace(rho) == 1]; % rho is positive and trace 1

% POVM constraitns
for k = 1:num_povms
    C = [C, trace((rho-inferred_rho)*povms(:,:,k)) == 0];
end
%%% ---------------------------------

% Set some options for YALMIP and solver
%options = sdpsettings('verbose',1,'solver','csdp');
options = sdpsettings('verbose',1);

% yalmip always minimizes objective
Objective = trace(O*rho);

% MINIMUM EXPECTATION
optimize(C, Objective, options);
bounds(1,1) = trace(O*value(rho));

% MAXIMUM EXPECTATION
optimize(C, -Objective, options);
bounds(1,2) = trace(O*value(rho));

end
