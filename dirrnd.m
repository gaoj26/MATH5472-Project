function theta = dirrnd(alpha)
% Dirichlet prior for word distribution
% draw a sample from a dirichlet prior with the parameter vector alpha
theta = randg(alpha);
theta = theta/sum(theta);
