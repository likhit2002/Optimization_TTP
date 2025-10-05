import xml.etree.ElementTree as ET

def get_distance_matrix(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    
    teams_section = root.find('.//Teams')
    n_teams = len(teams_section.findall('team'))
    
    distances_section = root.find('.//Distances')
    distances = [[0 for _ in range(n_teams)] for _ in range(n_teams)]
    for entry in distances_section.findall('distance'):
        i = int(entry.attrib['team1'])
        j = int(entry.attrib['team2'])
        d = int(entry.attrib['dist'])
        distances[i][j] = d
    
    
    return distances

distances = get_distance_matrix("C:/Users/Likhit/OneDrive/Desktop/Masters/Q1/Optimization of Data  Science/TTP/Data/NL6.xml")
for row in distances:
    print(row)