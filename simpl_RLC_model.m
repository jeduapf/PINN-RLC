%% Variables

R = 4.47213595499958;
c = 0.3;
L = 1.5;
R = 2*sqrt(L/c);

A = [ 0,      1 ;
    -1/(L*c), -R/L];

B = [   0  ;
     1/L];
    
C = [1, 0];

D = [0];

%%
sys = ss(A,B,C,D);
figure
bodemag(sys)