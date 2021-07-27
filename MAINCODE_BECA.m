%%%%%%%%%%Training a NARX Network%%%%%%
clear all;
E1 = 20

%Input file for each state
[num,txt,raw] = xlsread('West_Bengal.xlsx');
[num1,txt1,raw1] = xlsread('Westbengal_model_2.xlsx');


step_ahead1 = 366;
step_ahead = E1 + 366;
Y1 = num(1:end-E1,5)';
Y = con2seq(Y1);

%[num(end-E1+10:end,[1:3,4]); num1(10:step_ahead1,[1:3,10])]
% --------------------------------------------------------------------------------------------------
% Constructing the input Matrix
X1(1,:) = [num(end-E1:end,1); num1(1:step_ahead1,1)];
X1(2,:) = [num(end-E1:end,1); num1(1:step_ahead1,1)];
X1(3,:) = [num(end-E1:end,1); num1(1:step_ahead1,1)];
X1(4,:) = [num(end-E1:end,1); num1(1:step_ahead1,1)];
X1(5,:) = [num(end-E1:end,1); num1(1:step_ahead1,1)];

X2(1,:) = [num(end-E1:end,2); num1(1:step_ahead1,2)];
X2(2,:) = [num(end-E1:end,2); num1(1:step_ahead1,2)];
X2(3,:) = [num(end-E1:end,2); num1(1:step_ahead1,2)];
X2(4,:) = [num(end-E1:end,2); num1(1:step_ahead1,2)];
X2(5,:) = [num(end-E1:end,2); num1(1:step_ahead1,2)];


X3(1,:) = [num(end-E1:end,4); num1(1:step_ahead1,10)];
X3(2,:) = [num(end-E1:end,4); num1(1:step_ahead1,10)];
X3(3,:) = [num(end-E1:end,4); num1(1:step_ahead1,10)];
X3(4,:) = [num(end-E1:end,4); num1(1:step_ahead1,10)];
X3(5,:) = [num(end-E1:end,4); num1(1:step_ahead1,10)];

% ----------------------------------------------------------------------------------------------------------

X11(1,:) = num(10:end-E1,1)
X11(2,:) = num(7:end-E1-3,1);
X11(3,:) = num(5:end-E1-5,1);
X11(4,:) = num(3:end-E1-7,1);
X11(5,:) = num(1:end-E1-9,1);

X12(1,:) = num(10:end-E1,2)
X12(2,:) = num(7:end-E1-3,2);
X12(3,:) = num(5:end-E1-5,2);
X12(4,:) = num(3:end-E1-7,2);
X12(5,:) = num(1:end-E1-9,2);


X13(1,:) = num(10:end-E1,4)
X13(2,:) = num(7:end-E1-3,4);
X13(3,:) = num(5:end-E1-5,4);
X13(4,:) = num(3:end-E1-7,4);
X13(5,:) = num(1:end-E1-9,4);


In_1 = num1
D1 = num1(1:step_ahead1,11);
Num1 = [X1;X2;X3]
Num2 = [num(end-E1+1:end,5); num1(1:step_ahead1,11)]
Xf1 = [X1;X2;X3]';
Xpre =  tonndata(Xf1,false,false)
Yin = num(10:end-E1,5);
Xin = [X11;X12;X13]';
% 
% %Xpredf = tonndata(Xpred,false,false);

X = tonndata(Xin,false,false);
T = tonndata(Yin,false,false);
% 
trainFcn = 'trainscg';
inputDelays = 1;
feedbackDelays = 1;
hiddenLayerSize = 10;
a = 1;

Yiter = [];
for i = 1:1:200
    net = narxnet(inputDelays,feedbackDelays,hiddenLayerSize,'open',trainFcn);
    net.trainParam.showWindow = false;
    
    % Prepare the Data for Training and Simulation
    % The function PREPARETS prepares timeseries data for a particular network,
    % shifting time by the minimum amount to fill input states and layer
    % states. Using PREPARETS allows you to keep your original time series data
    % unchanged, while easily customizing it for networks with differing
    % numbers of delays, with open loop or closed loop feedback modes.
    [x,xi,ai,t] = preparets(net,X,{},T);
    net.divideParam.trainRatio = 83/100;
    net.divideParam.valRatio = 1/100;
    net.divideParam.testRatio = 16/100;
    
    net.performFcn = 'mse';  % Mean Squared Error
    
    % Choose Plot Functions
    % For a list of all plot functions type: help nnplot
    %net.plotFcns = {'plotperform','plottrainstate', 'ploterrhist', ...
    %    'plotregression', 'plotresponse', 'ploterrcorr', 'plotinerrcorr'};

    
    % Train the Network
    [net,tr] = train(net,x,t,xi,ai);
    y = net(x,xi,ai);
    performance = perform(net,t,y);
    if(performance < 1000)
        netc = closeloop(net);
        %netc.name = [net.name ' - Closed Loop'];
        %view(netc)
        [Xcs,Xci,Aci,Tcs] = preparets(netc,X,{},T);
        Ycs = netc( Xcs, Xci, Aci );
        [ net_new, tr, Ycs_new, Ecs, Xcf, Acf ] = train( netc, Xcs, Tcs, Xci, Aci );
        Ypre = ones(length(Xpre),1);
        Ypre = tonndata(Ypre,false,false);
        
        [xc,xic,aic,tc] = preparets(net_new,Xpre,{},Ypre);
        yc = netc(xc,xic,aic);

        colororder = {[0, 0.4470, 0.7410];
            [0.8500, 0.3250, 0.0980];
            [0.5, 0.5, 0.5]}
        
        y_final1 = cell2mat(yc);
        Yact(a,:) = cell2mat(y);
        Yiter(a,:) = y_final1;
        datecell2 = [raw(end-E1+1:end,1);raw1(2:end,1)];
        Aper(a) = performance;
        a = a + 1;

    end
end

Ap = min(Aper)
for i = 1:1:length(Aper)
    if(Aper(i) == Ap)
       yMean = Yiter(i,:); 
       Ytrain = Yact(i,:);
    end    
end

N = size(Yiter,1);
yMean = mean(Yiter);                                    % Mean Of All Experiments At Each Value Of ‘x’
ySEM = std(Yiter)/sqrt(N);                              % Compute ‘Standard Error Of The Mean’ Of All Experiments At Each Value Of ‘x’
CI95 = tinv([0.05 0.95], N-1);                    % Calculate 95% Probability Intervals Of t-Distribution
yCI95 = bsxfun(@times, ySEM, CI95(:));              % Calculate 95% Confidence Intervals Of All Experiments At Each Value Of ‘x’



datecell = raw(3:end-E1,1);
datecell2 = [raw(end-E1+1:end,1);raw1(2:end,1)];
p1 = plot(datetime(datecell2(1:step_ahead1)),yMean(end-366+1:end),'Linewidth',1,'color',colororder{2})
hold on
p2 = plot(datetime(datecell(1:step_ahead1-32)),Y1(1:step_ahead1-32),'Linewidth',1,'color',colororder{1})
hold on
plot(datetime(datecell2(length(feedbackDelays)+1:step_ahead)),Num2(length(feedbackDelays)+1:step_ahead),'Linewidth',1,'color',colororder{1})

xlabel({'Time'},'FontSize',12,'FontWeight','bold','Color','k')
ylabel({'ELectricity Consumption','(in GW/day)'},'FontSize',12,'FontWeight','bold','Color','k')
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'fontsize',12,'FontWeight','bold')
%legend([p2 p1],'Actual','Forecasted')

Yfo = Ytrain;
for i = 1:1:length(Yfo)
    if(isnan(Yfo(i)) ~= 1)
     error(i) =  (Yfo(i) - Y1(length(feedbackDelays)+i))^2
    end
end
rmse = sqrt(sum(error)/length(error));

tstart = datenum(2020,1,1)
tend = datenum(2020,12,31)
figure(11)
Ap1 = yMean + 1.96*rmse;
Ap2 = yMean - 1.96*rmse;

p11 = plot(datetime(tstart:tend,'ConvertFrom','datenum'),yMean(length(yMean)-366+1:end),'Linewidth',2,'color',colororder{1})
hold on

ha=shadedplot(datetime(tstart:tend,'ConvertFrom','datenum'),Ap1(length(Ap1)-366+1:end),Ap2(length(Ap1)-366+1:end), [0.5, 0.5, 0.5], 'w');
hold on
p21 = plot(datetime(tstart:tend,'ConvertFrom','datenum'),num1(1:366,11),'Linewidth',2,'color',colororder{2})
hold on
