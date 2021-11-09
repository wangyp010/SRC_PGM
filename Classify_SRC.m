function outLabel=Classify_SRC(A,Labels,x,y)
Number_Of_Classes = max(Labels);
res_y = zeros(Number_Of_Classes,size(x,2));
for tmpC1 = 1:Number_Of_Classes
    classIndex = (Labels==tmpC1);
    delta_x = zeros(size(x));
    delta_x(classIndex,:) = x(classIndex,:);
    res_y(tmpC1,:) = sum((y - A*delta_x).^2);
end
[~,MinIndex] = min(res_y);
outLabel = MinIndex;    
end