'''Parameters'''
import torch 
SEQ_LENGTH = 4
NEW_SEQ = 2
CUT = 15
device = torch.device("cpu")


'''Code Table'''
age_hierarchy = {
    0: "0-5",
    1: "5-10",
    2: "10-15",
    3: "15-20",
    4: "20-25",
    5: "25-30",
    6: "30-35",
    7: "35-40",
    8: "40-45",
    9: "45-50",
    10: "50-55",
    11: "55-60",
    12: "60-65",
    13: "65-70",
    14: "70-75",
    15: "75-80",
    16: "80-85",
    17: "85~"
}

occupation = {
    1: "Agriculture, Forestry and Fisheries",
    2: "Product Process and Labor Service",
    3: "Sales",
    4: "Service sector",
    5: "Traffic and communication service",
    6: "Security service",
    7: "Clerical job",
    8: "Professional and technical job",
    9: "Management staff",
    10: "Other occupation",
    11: "Kindergarten child, elementary school student, and middle school student",
    12: "High school student",
    13: "University student and vocational school student",
    14: "Housewife and househusband",
    15: "Inoccupation",
    16: "Others",
    91: "Unknown number",
    99: "Unknown"
}

trip_purpose = {
    1: "Commute to work",
    2: "Commute to school",
    3: "Return home",
    4: "Shopping",
    5: "Social and recreation",
    6: "Travel and leisure",
    7: "Hospital visit",
    8: "Other private business",
    9: "Pickup and drop-off",
    10: "Sales, delivery, purchase",
    11: "Meeting, bill collection",
    12: "Operation",
    13: "Work agriculture & fisheries",
    14: "Other works",
    99: "Others"
}

transport_type = {
    1: "Walk",
    2: "Bicycle",
    3: "Motorcycle",
    4: "Bike",
    5: "Taxi",
    6: "Vehicle",
    7: "Light vehicle",
    8: "Shipping vehicle",
    9: "Private or reserved bus",
    10: "Route or highway bus",
    11: "Monorail, new traffic",
    12: "Railway, subway",
    13: "Ship",
    14: "Airplane",
    15: "Others",
    97: "Stay",
    99: "Unknown"
}

landuse = {
    0: "Forest",
    1: "Grassland",
    2: "Rice Field",
    3: "Agricultural",
    4: "Industrial",
    5: "Urban area", 
    6: "Water land",
    7: "Others",
    8: "Ocean"
}
