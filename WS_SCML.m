% Weighted Signed Spectral Clustering of Multilayer networks (WS-SCML)

% Sema Athamnah, athamnah@msu.edu

% Step 1- Load SSBM data
load 'SSBM_data.mat'

% Step2- run WS-SCML
kc   = 5;
tol  = 1e-5;
max_iter = 1000;
L = size(SSBM_1,2);
N = size(SSBM_1,3);
alpha_range =0.1:0.1:1; 

rng(1);opts = statset('MaxIter', 200, 'UseParallel', false);
W0 = ones(L,1) / sqrt(L);  % initialized W0 such that ||w||2 = 1
[U_star_init,~] = qr(randn(N,kc),0); % initialized U_star - random orthonormal basis (N xkc)

% Preallocate
% W_all       = zeros(L, numel(alpha_range));
% U_star_all  = cell(numel(alpha_range),1);
% it_all      = zeros(numel(alpha_range),1);

fprintf('Start (Eq. 1)\n');
tic
for NL = 1:5 % noise level
    Data_level = eval(sprintf('SSBM_%d', NL));   
    for is=1:size(Data_level,1) % sample level

        D=squeeze(Data_level(is,:,:,:));
        A=permute(D, [2, 3, 1]);

for i=1:size(A,3)
    As=A(:,:,i);
    As(1:N+1:end) = 1; 
    A_new(:,:,i)=As;
end

Ap=A_new.*(A_new>0);
An=-A_new.*(A_new<0);

for i = 1:size(A_new,3)

    D_bar= diag(sum(abs(A_new(:,:,i)),2));
    D_barinv=D_bar^(-0.5);
    Apk=Ap(:,:,i);
    %Dpk=diag(sum(Apk,2));
    Ap_norm(:,:,i)= D_barinv * (Apk) * D_barinv;
    Ank = An(:,:,i);
    Dnk = diag(sum(Ank,2)); 
    %An_norm(:,:,i)=D_barinv * (Ank) * D_barinv;
    %Lp(:,:,i)=Dpk-Apk;
    Ln(:,:,i) = Dnk - Ank;
    Ln_norm(:,:,i) = D_barinv * (Dnk - Ank) * D_barinv;
end
for a = 1:numel(alpha_range)
    alpha_1 = alpha_range(a);

    % reset
    W = W0;
    U_star_old = U_star_init;
    iter = 0;
    converged = false;

    while ~converged && iter < max_iter
        iter = iter + 1;

        % ---- Solve for U (given W)
        % X = sum_l w_l (A^+_l + alpha_1 L^-_l)
        X = sum((Ap_norm + (alpha_1*Ln_norm)) .* reshape(W,1,1,[]), 3);
        % X = (X + X.')/2;% enforce symmetry for numerical stability
        [U_star, ~] = eigs(X, kc, 'largestreal');% top-kc eigenvectors
        U_star = U_star./sqrt(sum(U_star.^2, 2));

        % ---- Solve for W (given U_star)
        % c_l = tr(U_star' (Ap_l + alpha_1*Ln_l) U_star)
        c = zeros(L,1);
        for i = 1:L
            G = Ap_norm(:,:,i) + alpha_1*Ln_norm(:,:,i);
            c(i) = trace(U_star' * G * U_star);
        end

        % maximize c' * w s.t. w >= 0, ||w||_2 = 1 (valid both ways)
        if any(c > 0)
            cp = max(c,0);
            W  = cp / norm(cp,2);
        else
            [~, idx] = max(c);        % least-negative entry
            W = zeros(L,1); W(idx) = 1;
        end

        % ---- convergence (subspace distance via projection)
        change = norm(U_star*U_star' - U_star_old*U_star_old', 'fro');
        fprintf('alpha=%.3f  iter=%d  change=%.3e\n', alpha_1, iter, change);
        converged = (change < tol);
        U_star_old = U_star;
    end

    % save results for this alpha
    W_all(:,a) = W;
    U_star_all{a}   = U_star;
    it_all(a)  = iter; 
    [idx, ~] = kmeans(U_star, kc, 'Replicates', 20, 'Options', opts);
    NMI_values(a)=getNMI(idx,GT);
    ARI_values(a)=rand_index(idx,GT,'adjusted');
    if converged, fprintf('  -> Converged at iter %d.\n', iter); end
end

[max_nmi,id]=max(NMI_values);
optimal_nmi(is,NL)=max_nmi;
optimal_ari(is,NL)=ARI_values(id);
optimal_alpha(is,NL)=alpha_range(id);
%optimal_W=W_all(:,id);
%optimal_U=U_star_all{id};
    end
end
elapsed = toc;fprintf('Done in %.2f s.\n', elapsed);

figure,plot(mean(optimal_nmi)),
title('Mean NMI across 20 samples')
xlabel('Noise level'),ylabel('NMI'), grid on

figure,plot(mean(optimal_ari)),
title('Mean ARI across 20 samples')
xlabel('Noise level'),ylabel('ARI'), grid on

figure,imagesc(optimal_alpha),title('Optimal \alpha_1')
ylabel('Noise level'),ylabel('Samples'),colorbar