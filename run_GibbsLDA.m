% main program to run the demo of Gibbs sampling for LDA learning
clc;
clear;

%% generation of documents/images
gen_images;

%% run Gibbs sampling
[est_Phi, est_Theta, est_z, logPw_z, Phi_iter]=learn_GibbsLDA(w, K, 1, 1, 200, 150, 5);

%% visualize learned topics in different iterations
figure; colormap 'gray'
for iter=1:200
    if (mod(iter,10)==0)
        for k=1:8
            subplot(20,8,(iter/10-1)*8+k); imagesc(reshape(Phi_iter(:,k,iter), [4 4])); axis equal; axis tight;
            set(gca,'xtick',[])
            set(gca,'ytick',[])
        end
    end
end

print('-djpeg', 'Phi_iter.jpg');

%% plot log-likelihood
figure;
plot(logPw_z);
hold on;
plot(10:10:200,logPw_z(10:10:200),'o');
xlabel('iterations')
ylabel('log p(W|Z)')

print('-djpeg','logPw_z.jpg')

%% visualize initial estimated topics
figure; colormap 'gray'
for i=1:8
    subplot(1,8,i); imagesc(reshape(Phi_iter(:,i,1), [4 4])); axis equal; axis tight;
    set(gca,'xtick',[])
    set(gca,'ytick',[])
end
print('-djpeg', 'initial_Phi.jpg');

%% visualize final estimated topics
figure; colormap 'gray'
for i=1:8
    subplot(1,8,i); imagesc(reshape(est_Phi(:,i), [4 4])); axis equal; axis tight;
    set(gca,'xtick',[])
    set(gca,'ytick',[])
end
print('-djpeg', 'final_Phi.jpg');

save('param_est.mat','est_Phi','est_Theta','est_z','logPw_z','Phi_iter');
