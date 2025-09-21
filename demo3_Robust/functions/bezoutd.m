function [Rp,Sp,nrp,nsp]=bezoutd(A,B,Hs,Hr,P)
%function [Rp,Sp,nrp,nsp]=bezoutd(A,B,Hs,Hr,P)
%solves AHsSp+BHrRp=P by coefficient comparison. Delay and discretization delay need to be integrated in B 
%(zeros at the begining).
%
%inputs:
%A=[a0 a1 ... aNa] ... vector of model denominator coefficients A=a0 + a1z^(-1) + a2z^(-2) +...+ aNaz^(-Na) 
%B=[b0 b1 ... bNb] ... vector of model numerator coefficients B=b0 + b1z^(-1) + b2z^(-2) +...+ bNbz^(-Nb)
%Hs=[hs0 hs1 ... hsNhs] ... vector of controller denominator fixed part Hs=hs0 + hs1z^(-1) +...+ hsNhsz^(-Nhs) 
%Hr=[hr0 hr1 ... hrNhr] ... vector of controller denominator fixed part Hr=hr0 + hr1z^(-1) +...+ hrNhrz^(-Nhr) 
%P=[p0 p1 ... pNp] ... vector of desired polynomial coefficients P=p0 + p1z^(-1) + p2z^(-2) +...+ pNpz^(-Np) 
%outputs:
%Rp=[rp0 rp1 rp2 ...] ... vector of coefficients for resulted controller numerator 
%Sp=[sp0 sp1 sp2 ...] ... vector of coefficients for resulted controller denominator 
%nrp ... order of Rp
%nsp ... order of Sp
%
%written by: J. Langer, I.D. Landau, H. Prochazka
%7th june 2002
%Modified 06/08/2007 : "convz" replaced by "conv"


PRECISION=1e-16;

D=size(A);
if D(1)>1, A=A'; end;
D=size(B);
if D(1)>1, B=B'; end;
D=size(Hs);
if D(1)>1, Hs=Hs'; end;
if D(1)==0, Hs=1; end;
D=size(Hr);
if D(1)>1, Hr=Hr'; end;
if D(1)==0, Hr=1;end;
D=size(P);
if D(1)>1, P=P'; end;

na=length(A)-1;
nb=length(B)-1;
np=length(P)-1;
nhs=length(Hs)-1;
nhr=length(Hr)-1;

if (nhs>0), Ah=conv(A,Hs); else Ah=A*Hs; end; % Ah = A * Hs
nah=length(Ah) -1;
if (nhr>0), Bh=conv(B,Hr); else Bh=B*Hr; end;
nbh=length(Bh) -1;
if (np>nah+nbh-1), disp('Bezout error: too many poles');end;


% increase size of P using zeros if necessary
if (np<nah+nbh-1),
% 	P=[P zeros(1,nah+nbh-1-np)];
% set remaining poles onto a circle with radius rmin
	rootsPdes=roots(P);
	nextra=nah+nbh-1-np;
	rmin=1e-16;
	angle=[0:nextra-1]'/nextra*2*pi;
	j=sqrt(-1);
	rootsPextra=rmin*exp(j*angle);
	P=poly([rootsPdes;rootsPextra]);
	np=nah+nbh-1;
end;
% P,

nsp=nbh-1;
nrp=nah-1;

%matrix is smaller than vector PD
if (np>nah+nbh-1), 
   disp('Order of model denominator is too low! Add a polynom of higher order to Hs or Hr. ');
end;

% ns=nsp+nhs
% nr=nrp+nhr

M=[];
for j=1:nsp+1, 
	V=[];
	if (j>1), V=[V ; zeros(j-1,1)]; end;% zeros in front of Ah
	V=[V ; Ah'];% Ah
	if (j<=nsp), V=[V ; zeros(nsp+1-j,1)]; end;% zeros behind Ah
	if (length(V)~=nah+nbh), disp('bezoutb: error V'); end;
  	M=[M V]; % add one column to M
end;

for j=1:nrp+1, 
	V=[];
	if (j>1), V=[V ; zeros(j-1,1)]; end;
	V=[V ; Bh'];
	if (j<=nrp), V=[V ; zeros(nrp+1-j,1)]; end;
   if (length(V)~=nah+nbh), disp('bezoutb: error V'); end;
  	M=[M V];
end;

D=size(M);
if (D(1)~=nah+nbh), disp('bezoutb: error size M row'); end;
if (D(2)~=nah+nbh), disp('bezoutb: error size M column'); end;

% make P column vector
P=P';


global M1;
M1=M;
      
X= M\P;
% coefficients are real values
X=real(X);

% make X row vector
X=X';
Sp=X(1:nsp+1);
Rp=X(nsp+2:nsp+nrp+2);
