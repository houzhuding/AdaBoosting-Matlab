%% Adaboosting Algorithm %%
%%      Houzhu Ding      %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [weak_learner,alpha_,accuracy]= AdaBoosting_ding_(data,round,fea_sel)
%% Initialize parameters
% sample numbers
d_length = size(data,1);m = d_length;
% fea_sel  = 1:size(data,2);
% Interation round of Adaboosting
T = round;
% define y  has -1 and 1 as label for mathematical operation
y = data(:,9);
k = find(data(:,9)==0);
y(k) = -1;
% thresholds in different axe at different round
theta = zeros(T,length(fea_sel));
% weak learner weight
alpha_ = zeros(T,1);
% Initilize D1(i) = 1 /m
D = zeros(1,m)+1/m;
%% Start Training AdaBoosting Loop 

for t = 1:T
   min_error_final = inf;
   for fea_idx = fea_sel
       % define step size and the threshold along certain axis
       step_size = (max(data(:,fea_idx))-min(data(:,fea_idx)))/m;
       theta_range = min(data(:,fea_idx)):step_size:max(data(:,fea_idx));
       % Find the best theta to minimize the error rate for one axis
       min_error = inf;
       for theta_idx = 1:length(theta_range)
            theta_current = theta_range(theta_idx);
            % Compute the error rate of current theta of two directions
            for direction = 1:2
                if direction == 1
                    data_pos1_class = data(:,fea_idx) >  theta_current; % class to 0
                else
                    data_pos1_class = data(:,fea_idx) <  theta_current; % class to 1
                end
                % Compute the missclassified (The label is denoted as 0 and 1
                error_classified = xor(data_pos1_class,data(:,9));
                error_ = D*error_classified;
                error_ = error_/sum(D);
                
                % Record the minimal error rate, with corresponding
                % threshold and direction
                if error_ < min_error
                    min_error = error_;
                    theta(t,fea_idx) = theta_current;
                    labelpos1_class = data_pos1_class;
                    dir = direction;
                end
            end   
       end
       if min_error < min_error_final
           min_error_final = min_error;
           labelpos1_class_fea = labelpos1_class;
           % define and assign weak learner parameters
           ht(t) = hypothesis_ding;
           ht(t).feature_idx = fea_idx;
           ht(t).direction = dir;
           ht(t).threshold = theta(t,fea_idx);
           ht(t).round = t;
       end
   end
   % Change zero labels to -1 labels
   zero_element = find(labelpos1_class_fea == 0);
   h = +labelpos1_class_fea;
   h(zero_element) = -1;
   h_out(t,:) = h;
   % Store the minimum error rate at round t
   eps(t) = min_error_final;
   % Coumpute the weight of weak learner t
   alpha_(t) = 0.5*log((1-eps(t))/eps(t));
   % Update the weight
   Zt =  0;
   for i = 1:m
        Zt = Zt+D(i)*exp(-alpha_(t)*y(i)*h_out(t,i));
   end
   for i = 1:m
        D(i) = D(i)*exp(-alpha_(t)*y(i)*h_out(t,i))/Zt;
   end
end

%% Assemble the AdaBoosting strong classifier and test accuracy
weak_learner = ht;
correct = 0;
for test_idx = 1:m
    y_predict = H_strong(ht,alpha_,data(test_idx,:),T);
    if y_predict == y(test_idx)
        correct = correct + 1;
    end
    label_predict(test_idx) = y_predict;
end
accuracy = correct/m;

% if length(fea_sel) > 1
%     plot_data = data;
%     plot_sample(plot_data,label_predict,fea_sel,weak_learner,T)
% end

function y = H_strong(weak_learner,alpha_,x,round)
ht = weak_learner;
H = 0;
for i = 1:round
    ht(i).input_x = x;
    H = H + alpha_(i)*G_weak_learner(ht(i));
end
y = sign(H);


% function plot_sample(data,label,fea_sel,weak_learner,T)
% 
% d_length = size(data,1);
% labelneg1 = data(:,9)== 0;
% labelpos1 = data(:,9)== 1;
% figure
% subplot(1,2,1)
% for i = 1:d_length
%     if labelpos1(i) == 1
%         plot(data(i,fea_sel(1)),data(i,weak_learner(T).feature_idx),'r*');
%         hold on
%     else
%         plot(data(i,fea_sel(1)),data(i,weak_learner(T).feature_idx),'b*');
%         hold on
%     end
% end
% 
% subplot(1,2,2)
% for i = 1:d_length
%     if label(i) == 1
%         plot(data(i,fea_sel(1)),data(i,weak_learner(T).feature_idx),'r*');
%         hold on
%     else
%         plot(data(i,fea_sel(1)),data(i,weak_learner(T).feature_idx),'bo');
%         hold on
%     end
% end