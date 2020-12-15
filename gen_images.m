%%% MATH5472 project Jing GAO

close all; clear;

% setting ground truth of topic-specific word distribution, Phi
% the size of the vocabulary is 16=4x4
% each position represents a unique word
% 8 topics in total

Phi(:,1) = [1 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0]/4; % word distribution under topic 1
Phi(:,2) = [0 1 0 0 0 1 0 0 0 1 0 0 0 1 0 0]/4; % word distribution under topic 2
Phi(:,3) = [0 0 1 0 0 0 1 0 0 0 1 0 0 0 1 0]/4; % word distribution under topic 5
Phi(:,4) = [0 0 0 1 0 0 0 1 0 0 0 1 0 0 0 1]/4; % word distribution under topic 4
Phi(:,5) = [1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0]/4; % word distribution under topic 5
Phi(:,6) = [0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0]/4; % word distribution under topic 6
Phi(:,7) = [0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0]/4; % word distribution under topic 7
Phi(:,8) = [0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1]/4; % word distribution under topic 8

% show the ground truth of Phi
figure; colormap 'gray'
for i=1:8
    subplot(1,8,i); imagesc(reshape(Phi(:,i), [4 4])); axis equal; axis tight;
    set(gca,'xtick',[])
    set(gca,'ytick',[])
end
print('-djpeg', 'Phi_ground_truth.jpg');

% hyperparameters of symmetric Dirichlet priors
alpha=1;

M=500; % the number of documents in corpus
K=8; % the number of latent topics
Nd=100; % the number of words in each document
V = 16; % the size of the vocabulary

% ground truth of document-specific topic distribution, Theta
z = zeros(M,Nd);
w = zeros(M,Nd);
for m=1:M
    theta = dirrnd(ones(1,K)*alpha); % draw a topic distribution for each document from the symmetric Dirichlet prior
    for n=1:Nd 
        z(m,n) = find(mnrnd(1,theta)==1); % draw a topic for each word from the document-specific topic distribution theta
        w(m,n) = find(mnrnd(1,reshape(Phi(:,z(m,n)),[1 V]))==1); % draw each word from the topic-specific word distribution Phi(:,z(m,n))
    end
    word_hist(:,:,m) = reshape(hist(w(m,:),16), [4 4]); % reconstruct the document to an image
    Theta(m,:) = theta; % document-specific topic distribution
end

% show the first 16 documents/images
figure; colormap 'gray'
for i=1:16
    subplot(4,4,i);
    imagesc(word_hist(:,:,i)); axis equal; axis tight;
    set(gca,'xtick',[])
    set(gca,'ytick',[])
end
print('-djpeg', 'example_documents.jpg');

save('demo_data.mat','word_hist','w','z','Theta','Phi','K', 'alpha');
