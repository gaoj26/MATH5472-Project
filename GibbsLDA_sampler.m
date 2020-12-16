function [Phi, Theta, est_z, logPw_z,Phi_iter]=GibbsLDA_sampler(w, K, alpha, beta, max_iter, ...
    burn_in_iter, sampling_lag)

%% initialization
disp('***initialization***');
[M,Nd] = size(w); % w is the corpus
V = length(unique(w));
NWZ = zeros(V,K)+beta;  % K by V Topic-Word count matrix 
NZM = zeros(K,M)+alpha; % K by M Document-Topic count matrix
NZ = sum(NWZ);			
z = zeros(M,Nd);		% topic assignmed to each word in the corpus	
Phi = zeros(V,K);		% parameters of word distributions need to be estimated
Theta = zeros(K,M);     % parameters of topic distributions need to estimated

% initialization by randomly assigning a topic to each word with identical
% probability

for m=1:M % for each document
    for n=1:Nd % for each word
        z(m,n) = find(mnrnd(1,ones(1,K)/K )==1); % draw a topic for each word in the corpus
        NZM(z(m,n),m) = NZM(z(m,n),m) + 1; % initialization of document-topic count matrix
        NWZ(w(m,n),z(m,n)) = NWZ(w(m,n),z(m,n)) + 1; % initialization of topic-word count matrix
        NZ(z(m,n)) = NZ(z(m,n)) + 1; % summation of one specific column/(row in the derivation) in NWZ
    end
end

% initialization of Phi
for k=1:K  
    Phi(:,k) = NWZ(:,k)/NZ(k); % the word distribution under a specific topic
end

for mm = 1:M
    Theta(:,mm) = NZM(:,mm)/sum(NZM(:,mm)); % the topic distribution of a specific document
end
    

%% Gibbs sampling
disp('Gibbs sampling');

% read_out_Phi and read_out_Theta store the sum of read-out Phi and Theta
read_out_Phi = zeros(V,K);
read_out_Theta = zeros(K,M);
read_out_sampling_num = 0;
logPw_z = zeros(1,max_iter);
Phi_iter = zeros(V,K,max_iter);

for iter = 1:max_iter
    for m=1:M % for each document
        for n=1:Nd % for each word
            % decrease three counts
            NZM(z(m,n),m) = NZM(z(m,n),m) - 1;
            NWZ(w(m,n),z(m,n)) = NWZ(w(m,n),z(m,n)) - 1;
            NZ(z(m,n)) = NZ(z(m,n)) -1;
            % update the posterior distribution of z, p(z_i)
            p=zeros(1,K);
            for k=1:K
                p(k) = NWZ(w(m,n),k)/NZ(k) * NZM(k,m);
            end
            p = p/sum(p); % normalization
            % draw topic for this word
            z(m,n) = find(mnrnd(1,p)==1); 
            % increase three counts
            NZM(z(m,n),m) = NZM(z(m,n),m) + 1;
            NWZ(w(m,n),z(m,n)) = NWZ(w(m,n),z(m,n)) + 1;
            NZ(z(m,n)) = NZ(z(m,n)) + 1;
        end
    end
    
    loghood=0;
    for k=1:K
        Phi_iter(:,k,iter)=NWZ(:,k)/NZ(k);
        for v=1:V
            loghood=loghood+NWZ(v,k)*log(NWZ(v,k)/NZ(k));
        end
    end
    logPw_z(iter)=loghood;
   
    if ((mod(iter,sampling_lag) == 0) || (iter == 1))
        if iter >= burn_in_iter % read out parameters after burn-in
            read_out_sampling_num = read_out_sampling_num + 1;
            for k=1:K
                read_out_Phi(:,k) = read_out_Phi(:,k) + NWZ(:,k)/NZ(k);
            end
            for mm = 1:M
                read_out_Theta(:,mm) = read_out_Theta(:,mm) + NZM(:,mm)/sum(NZM(:,mm));
            end
        end
    end
end

% finally, parameters are obtained by averaging the read-out values computed from
% the samples after the burn-in period
Phi = read_out_Phi/read_out_sampling_num;
Theta = read_out_Theta/read_out_sampling_num;
est_z=z;



