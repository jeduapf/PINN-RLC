%% Variables

R = 1.2;
L = 1.5;
c = 0.3;

A = [ 0,      1 ;
    -1/(L*c), -R/L];

B = [   0  ;
     1/(L*c)];
    
C = [1, 0];

D = [0];

%%
sys = ss(A,B,C,D);
figure
bodemag(sys)