data = jsondecode(fileread('Hungary_major_connections_simple.json'));
%Likely there are better ways, may just import the locations directly
location_pairs = fieldnames(data);

%LUT
lookup_table = containers.Map(location_pairs, 1:length(location_pairs));

% Create a vector of all road minimum traffic values
min_traffic_values = [];
for i = 1:length(location_pairs)
    roads = data.(location_pairs{i});
    for j = 1:length(roads)
        min_traffic_values = [min_traffic_values; roads(j).min_traffic];
    end
end

%Basic P matrix
P = zeros(length(min_traffic_values), length(location_pairs));
row = 1;
for i = 1:length(location_pairs)
    roads = data.(location_pairs{i});
    for j = 1:length(roads)
        P(row, i) = 1;
        row = row + 1;
    end
end