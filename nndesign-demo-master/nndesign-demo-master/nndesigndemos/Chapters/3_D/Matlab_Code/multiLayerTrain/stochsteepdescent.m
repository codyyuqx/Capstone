function stochsteepdescent()

figure(1)

hh=[-1 2 0 -1;2 -1 -1 0];
t = [-1;-1;1;1];
jj = hh*hh';
jt = hh*t;

a=2*jj;b=-2*jt;c=t'*t;

flag=1;
while(flag~=0)

tt=(.01:.01:1)'*2*pi;
circ_x1=sin(tt)*.01*3/2;
circ_y1=cos(tt)*.01*3/2;
circ_x2=sin(tt)*.02*3/2;
circ_y2=cos(tt)*.02*3/2;

% a=[10 -6;-6 10];
% b=[4;4];
% c=0;

x10=-1;
x20=-2.95;
% x20=-1;



clf
hold off

%disp('Pick Values with Mouse/Arrow Keys');


% PLOT ERROR CONTOUR
%===================

%y=-2.5:.05:-0.5;x=y+1;
y=-3:.05:0;x=y;
% y=-2:.05:-1;x=y;
[X1,X2]=meshgrid(x,y);
F = (a(1,1)*X1.^2 + (a(1,2)+a(2,1))*X1.*X2 + a(2,2)*X2.^2)/2 ...
       + b(1)*X1 + b(2)*X2 +c;

clf reset
cont=[0.5 1 2 4 8 16 32];
ch = contour(x,y,F,cont);


set(gcf,'inverthardcopy','off')

set(gca,'XColor',[0 0 0]);
set(gca,'YColor',[0 0 0]);
set(gcf,'Color',[1 1 1]);
set(gca,'xtick',[-3 -2 -1 0])
set(gca,'ytick',[-3 -2 -1 0])
% set(gca,'xtick',[-2 -1.5 -1])
% set(gca,'ytick',[-2 -1.5 -1])

%  convert contour colors to grey scale

ch1=get(gca,'Children');
ch = get(ch1,'Children');
%ch=get(gca,'Children');
for i=1:length(ch),
  %col = set(ch(i),'Color','k');
  set(ch(i),'EdgeColor',zeros(1,3));

end

xlabel('$w_{1,1}$','Interpreter','LaTex','FontSize', 16);
ylabel('$w_{1,2}$','Interpreter','LaTex','FontSize', 16);

axis('square')

hold on

%  add eigenvectors

%[ev,eval]=eig(a);
%sol=-pinv(a)*b;
%nndrwvec([sol(1) sol(1)+ev(1,1)],[sol(2) sol(2)+ev(2,1)],2,.08,[0 0 1]);
%nndrwvec([sol(1) sol(1)+ev(1,2)],[sol(2) sol(2)+ev(2,2)],2,.08,[0 0 1]);


%[x10,x20] = ginput(1);
xc=circ_x2+x10;
yc=circ_y2+x20;
%fill(xc,yc,'b','erasemode','none','EdgeColor','b')
fill(xc,yc,'b','EdgeColor','b')


% TRAINING PARAMETERS
disp_freq = 1;
max_epoch = 400;
err_goal = -3.999;

% INITIALIZE PARAMETERS
x1 = x10;
x2 = x20;


lr = 0.01;

for epoch=1:max_epoch

  % CHECK PHASE
%   if SSE < err_goal, epoch=epoch-1; break, end

  Lx1 = x1; Lx2 = x2;

  % LEARNING PHASE
  select = randperm(4);
  select = select(1);
  p = hh(:,select);
  e = t(select)-[x1 x2]*p;
  grad=2*e*p;
  grad1=a*[x1;x2]+b;


  dx1=lr*grad(1);
  dx2=lr*grad(2);

  x1 = x1 + dx1;
  x2 = x2 + dx2;

  SSE = (a(1,1)*x1.^2 + (a(1,2)+a(2,1))*x1.*x2 + a(2,2)*x2.^2)/2 ...
       + b(1)*x1 + b(2)*x2 +c;

  % DISPLAY PROGRESS
  if rem(epoch,disp_freq) == 0
    xc=circ_x1+x1;
    yc=circ_y1+x2;
    %fill(xc,yc,'b','erasemode','none','EdgeColor','b')
    fill(xc,yc,'b','EdgeColor','b')
    
    %plot([Lx1 x1],[Lx2 x2],'b-','linewidth',2,'erasemode','none');
    plot([Lx1 x1],[Lx2 x2],'b-','linewidth',2);
    end
  end

% PLOT FINAL VALUES
%==================

% if SSE < err_goal,
%   xc=circ_x2+x1;
%   yc=circ_y2+x2;
%   fill(xc,yc,'w','erasemode','none','EdgeColor','b')
% end

%   z = menu('Make Your Choice', ...
%     'Start Over', ...
%     'Quit');
%   disp('');

%   if z == 2
    flag=0;
%   end

% hold on
% quiver(X1,X2,-2*dxx1,-2*dxx2,'k')

end

%print -depsc -tiff -r600 StochSteepDescent


