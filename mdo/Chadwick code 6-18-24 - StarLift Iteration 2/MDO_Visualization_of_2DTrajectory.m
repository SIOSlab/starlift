%Arthur Chadwick
%StarLift MDO Visualization of Trajectory
%5/7/2024
clear all
close all
clc

for trajectory_module = 1:1

I = imread("Earth.png");
image('CData',I,'XData',[-1 1],'YData',[-1 1])
hold on

axis square
axis off

rE = 0.2; %[m] radius of Earth

i = 1;
t(i) = 0; %[s]
x(i) = 0; %[m]
y(i) = 2; %[m]
r(i) = sqrt(x(i)^2 + y(i)^2); %[m]
vx(i) = 1; %[m/s]
vy(i) = 0; %[m/s]
v(i) = sqrt(vx(i)^2 + vy(i)^2); %[m/s]
a(i) = 10/((1 + r(i)/rE)^2); %[m/s^2]
theta_a(i) = atand(x(i)/y(i)); %[deg]
ax(i) = -a(i)*sind(theta_a(i)); %[m/s^2]
ay(i) = -a(i)*cosd(theta_a(i)); %[m/s^2]
plot(x(i),y(i),'*')

delta = 1000;

for i = 2:delta
    t(i) = t(i-1) + 100*0.5/delta; %[s]
    x(i) = x(i-1) + vx(i-1)*(t(i)-t(i-1)) + (1/2)*ax(i-1)*((t(i)-t(i-1))^2); %[m]
    y(i) = y(i-1) + vy(i-1)*(t(i)-t(i-1)) + (1/2)*ay(i-1)*((t(i)-t(i-1))^2); %[m]
    r(i) = sqrt(x(i)^2 + y(i)^2); %[m]
    a(i) = 0; %[m/s^2]
    if y(i) > 0 && x(i) > 0 %Quadrant I
        theta_a(i) = atand(x(i)/y(i)); %[deg]
        ax(i) = -a(i)*sind(theta_a(i)); %[m/s^2]
        ay(i) = -a(i)*cosd(theta_a(i)); %[m/s^2]
        v(i) = v(i-1) + a(i)*(t(i)-t(i-1)); %[m/s]
        theta_v(i) = 90 - theta_a(i); %[deg]
        vx(i) = v(i)*sind(theta_v(i)); %[m/s]
        vy(i) = -v(i)*cosd(theta_v(i)); %[m/s]
    elseif y(i) < 0 && x(i) > 0 %Quadrant II     
        theta_a(i) = atand(x(i)/-y(i)); %[deg]
        ax(i) = -a(i)*sind(theta_a(i)); %[m/s^2]
        ay(i) = a(i)*cosd(theta_a(i)); %[m/s^2]
        v(i) = v(i-1) + a(i)*(t(i)-t(i-1)); %[m/s]
        theta_v(i) = 90 - theta_a(i); %[deg]
        vx(i) = -v(i)*sind(theta_v(i)); %[m/s]
        vy(i) = -v(i)*cosd(theta_v(i)); %[m/s]
    elseif y(i) < 0 && x(i) < 0 %Quadrant III
        theta_a(i) = atand(-x(i)/-y(i)); %[deg]
        ax(i) = a(i)*sind(theta_a(i)); %[m/s^2]
        ay(i) = a(i)*cosd(theta_a(i)); %[m/s^2]
        v(i) = v(i-1) + a(i)*(t(i)-t(i-1)); %[m/s]
        theta_v(i) = 90 - theta_a(i); %[deg]
        vx(i) = -v(i)*sind(theta_v(i)); %[m/s]
        vy(i) = v(i)*cosd(theta_v(i)); %[m/s]
    elseif y(i) > 0 && x(i) < 0 %Quadrant IV
        theta_a(i) = atand(-x(i)/y(i)); %[deg]
        ax(i) = a(i)*sind(theta_a(i)); %[m/s^2]
        ay(i) = -a(i)*cosd(theta_a(i)); %[m/s^2]
        v(i) = v(i-1) + a(i)*(t(i)-t(i-1)); %[m/s]
        theta_v(i) = 90 - theta_a(i); %[deg]
        vx(i) = v(i)*sind(theta_v(i)); %[m/s]
        vy(i) = v(i)*cosd(theta_v(i)); %[m/s]
    end
    plot(x(i),y(i),'*')
    hold on
end

end























% t = 0:pi/100:16*pi;
% x1 = cos(t);
% y1 = sin(t); %// trajectory of object 1
% x2 = 2*cos(t);
% y2 = 2*sin(t); %// trajectory of object 2
% plot(x1,y1,'color',[.5 .5 .5]); %// plot trajectory of object 1
% hold on
% plot(x2,y2,'color',[.5 .5 .5]); %// plot trajectory of object 2
% h1 = plot(x1(1),y1(1),'ro'); %// plot initial position of object 1
% h2 = plot(x2(1),y2(1),'b*'); %// plot initial position of object 2
% axis([-2 2 -2 2]) %// freeze axis size
% grid on
% for n = 1:numel(t)
%     set(h1, 'XData', x1(n), 'YData', y1(n)); %// update position of object 2
%     set(h2, 'XData', x2(n), 'YData', y2(n)); %// update position of object 2
%     drawnow %// refresh figure
% end