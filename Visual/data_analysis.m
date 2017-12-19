clc
clear
close all
A=load('lx4 - Re300 - Fr300 - T2 - results.txt');
B=load('lx4 - Re300 - Fr300 - T2 - convergence.txt');
path1='E:\Direct PhD\Courses\CFD\project 2\p2\p2\lx2 - Re300 - Fr300';
path2='E:\Direct PhD\Courses\CFD\project 2\p2\p2\lx2 - Re300 - Fr300 - noconvection';
path3='E:\Direct PhD\Courses\CFD\project 2\p2\p2\lx4 - Re300 - Fr300';

s=size(A);
s2=size(B);
tt=B(:,2);pcounter=B(:,3);presidual=B(:,4);tv=B(:,5);
lx=4;ly=5;
nx=10*lx;ny=10*ly;
Re=10;j=1;
dx=lx/(nx-1);
dy=ly/(ny-1);
xx=0:dx:lx;yy=0:dy:ly;
t=A(:,1);x=A(:,2);y=A(:,3);u=A(:,4);v=A(:,5);p=A(:,6);j=1;T=A(:,7);om=A(:,8);
for i=1:length(t)
    if t(i)==min(t)     
        x5(j)=x(i);
        y5(j)=y(i);
        u5(j)=u(i);
        v5(j)=v(i);
        p5(j)=p(i);
        om5(j)=om(i);
        T5(j)=T(i);
        j=j+1;
    end
end

set(0,'DefaultAxesFontSize',18)
fig1=figure('units','normalized','outerposition',[0 0 1 1]);
subplot(3,1,1)
semilogx(tt,pcounter,'linewidth',2)
xlabel('time (s)')
ylabel('P. iterations')
title(['Pressure iterations versus time'])
axis tight
subplot(3,1,2)
loglog(tt,presidual,'linewidth',2)
xlabel('time (s)')
ylabel('P. residual')
title(['Pressure residual versus time'])
axis tight
subplot(3,1,3)
semilogx(tt,tv,'linewidth',2)
xlabel('time (s)')
ylabel('T. variance')
title('Total variance')
axis tight
j=1;
for i=1:length(t)
    if t(i)==max(t)     
        x5(j)=x(i);
        y5(j)=y(i);
        u5(j)=u(i);
        v5(j)=v(i);
        p5(j)=p(i);
        om5(j)=om(i);
        T5(j)=T(i);
        j=j+1;
    end
end

% Regenerate results with inlet and outlet
lim=dx*10;
xed=-lim:dx:0;xed2=lx:dx:lx+lim;
sf=length(xed)-1;
ushape=reshape(u5,nx,ny)';
vshape=reshape(v5,nx,ny)';
pshape=reshape(p5,nx,ny)';
omshape=reshape(om5,nx,ny)';
Tshape=reshape(T5,nx,ny)';

uinlet=zeros(ny,length(xed));
vinlet=zeros(ny,length(xed));
pinlet=zeros(ny,length(xed));
ominlet=zeros(ny,length(xed));
Tinlet=zeros(ny,length(xed));

uoutlet=zeros(ny,length(xed));
voutlet=zeros(ny,length(xed));
poutlet=zeros(ny,length(xed));
omoutlet=zeros(ny,length(xed));
Toutlet=zeros(ny,length(xed));

for i=1:length(xed)
uinlet(:,i)=ushape(:,1);
vinlet(:,i)=vshape(:,1);
pinlet(:,i)=pshape(:,1);
ominlet(:,i)=omshape(:,1);
Tinlet(:,i)=Tshape(:,1);

uoutlet(:,i)=ushape(:,nx);
voutlet(:,i)=vshape(:,nx);
poutlet(:,i)=pshape(:,nx);
omoutlet(:,i)=omshape(:,nx);
Toutlet(:,i)=Tshape(:,nx);

end
unew=zeros(ny,nx+2*sf);
vnew=zeros(ny,nx+2*sf);
pnew=zeros(ny,nx+2*sf);
omnew=zeros(ny,nx+2*sf);
Tnew=zeros(ny,nx+2*sf);
xnew=-lim:dx:lx+lim;
xb1=min(xnew):0.1:0;
yb1=0:0.1:2;
xb2=lx:0.1:max(xnew);
cc=zeros(length(yb1),length(xb1));

unew(:,1:sf)=uinlet(:,1:sf);
vnew(:,1:sf)=vinlet(:,1:sf);
pnew(:,1:sf)=pinlet(:,1:sf);
omnew(:,1:sf)=ominlet(:,1:sf);
Tnew(:,1:sf)=Tinlet(:,1:sf);
unew(:,nx+sf+1:nx+2*sf)=uoutlet(:,1:sf);
vnew(:,nx+sf+1:nx+2*sf)=voutlet(:,1:sf);
pnew(:,nx+sf+1:nx+2*sf)=poutlet(:,1:sf);
omnew(:,nx+sf+1:nx+2*sf)=omoutlet(:,1:sf);
Tnew(:,nx+sf+1:nx+2*sf)=Toutlet(:,1:sf);
unew(:,sf+1:nx+sf)=ushape(:,1:nx);
vnew(:,sf+1:nx+sf)=vshape(:,1:nx);
pnew(:,sf+1:nx+sf)=pshape(:,1:nx);
omnew(:,sf+1:nx+sf)=omshape(:,1:nx);
Tnew(:,sf+1:nx+sf)=Tshape(:,1:nx);

s=size(ushape);
fig2=figure('units','normalized','outerposition',[0 0 1 1]);
subplot(1,2,1)
pcolor(xnew,yy,unew)
shading('interp')
colorbar
xlabel('x')
ylabel('y')
title(['U velocity at ',' t ~ ',num2str(round(max(t))),'(s)'])
axis image
hold on
h = fill([-lim -lim 0 0],[0 2 2 0],'w');%axis([-2 2 -2 2]);
set(h,'edgecolor','black','facecolor',[0.73 0.83 0.96]);
h = fill([lx lx lx+lim lx+lim],[0 2 2 0],'w');%axis([-2 2 -2 2]);
set(h,'edgecolor','black','facecolor',[0.73 0.83 0.96]);

subplot(1,2,2)
pcolor(xnew,yy,vnew)
shading('interp')
colorbar
xlabel('x')
ylabel('y')
title(['V velocity at ',' t ~ ',num2str(round(max(t))),'(s)'])
axis image
hold on
h = fill([-lim -lim 0 0],[0 2 2 0],'w');%axis([-2 2 -2 2]);
set(h,'edgecolor','black','facecolor',[0.73 0.83 0.96]);
h = fill([lx lx lx+lim lx+lim],[0 2 2 0],'w');%axis([-2 2 -2 2]);
set(h,'edgecolor','black','facecolor',[0.73 0.83 0.96]);

fig3=figure('units','normalized','outerposition',[0 0 1 1]);
subplot(1,2,1)
pcolor(xnew,yy,omnew)
shading('interp')
colorbar
xlabel('x')
ylabel('y')
title(['Vorticity at ',' t ~ ',num2str(round(max(t))),'(s)'])
axis image
hold on
h = fill([-lim -lim 0 0],[0 2 2 0],'w');%axis([-2 2 -2 2]);
set(h,'edgecolor','black','facecolor',[0.73 0.83 0.96]);
h = fill([lx lx lx+lim lx+lim],[0 2 2 0],'w');%axis([-2 2 -2 2]);
set(h,'edgecolor','black','facecolor',[0.73 0.83 0.96]);

subplot(1,2,2)
pcolor(xnew,yy,Tnew)
shading('interp')
kk=colorbar;
title(kk,'(C)')
xlabel('x')
ylabel('y')
title(['Temp. at ',' t ~ ',num2str(round(max(t))),'(s)'])
axis image
hold on
h = fill([-lim -lim 0 0],[0 2 2 0],'w');%axis([-2 2 -2 2]);
set(h,'edgecolor','black','facecolor',[0.73 0.83 0.96]);
h = fill([lx lx lx+lim lx+lim],[0 2 2 0],'w');%axis([-2 2 -2 2]);
set(h,'edgecolor','black','facecolor',[0.73 0.83 0.96]);
%k222 = figure('Position', [100, 100, 1000, 800]);
fig4=figure('units','normalized','outerposition',[0 0 1 1]);
subplot(1,2,1)
scale_factor = 0.05*3;
quiver(xnew,yy,unew*scale_factor,vnew*scale_factor,'AutoScale','off')
colormap hsv
title(['Velocity vectors at ',' t ~ ',num2str(round(max(t))),'(s)'])
xlabel('x')
ylabel('y')
hold on
h = fill([-lim -lim 0 0],[0 2 2 0],'w');%axis([-2 2 -2 2]);
set(h,'edgecolor','black','facecolor',[0.73 0.83 0.96]);
h = fill([lx lx lx+lim lx+lim],[0 2 2 0],'w');%axis([-2 2 -2 2]);
set(h,'edgecolor','black','facecolor',[0.73 0.83 0.96]);
axis image
axis([min(xnew) max(xnew) min(yy) max(yy)])

subplot(1,2,2)
[sx,sy]=meshgrid(min(xnew):dx*3:max(xnew),min(yy):dy*3:max(yy));
h=streamline(xnew,yy,unew,vnew,sx,sy,[0.3 1000]);
title(['Streamlines at ',' t ~ ',num2str(round(max(t))),'(s)'])
set(h,'LineWidth',0.1,'Color','k');
xlabel('x')
ylabel('y')
hold on
h = fill([-lim -lim 0 0],[0 2 2 0],'w');%axis([-2 2 -2 2]);
set(h,'edgecolor','black','facecolor',[0.73 0.83 0.96]);
h = fill([lx lx lx+lim lx+lim],[0 2 2 0],'w');%axis([-2 2 -2 2]);
set(h,'edgecolor','black','facecolor',[0.73 0.83 0.96]);
axis image
saveas(fig1, 'fig1','jpg')
saveas(fig2, 'fig2','jpg')
saveas(fig3, 'fig3','jpg')
saveas(fig4, 'fig4','jpg')

