first iteration of a finite element simulation on a rectangle

using arbitary degree bernstein polynomials and analytical integration for computation of stiffnes matrix factors

makefile has yet to be tested *missing include of cusparse library*

to do:
- include dirichlet boundary conditions apliance fpr CSR type matrix (working for degree=1) on higher degree there are inconsistencies in the appliance of boundary conditions 
Example n=2 ElementsX=5 ElementsY=4

- computation crashes n=1 ElementsX*ElementsY>=282*282 -> watchdog shader termination?