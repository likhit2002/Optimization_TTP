import xml.etree.ElementTree as ET

def parse_xml(filename):
    tree = ET.parse(filename)
    root = tree.getroot()

    teams_section = root.find('.//Teams')
    team_names = [team.attrib['name'] for team in teams_section.findall('team')]
    n_teams = len(team_names)

    distances_section = root.find('.//Distances')
    distances = [[0 for _ in range(n_teams)] for _ in range(n_teams)]
    for entry in distances_section.findall('distance'):
        i = int(entry.attrib['team1'])
        j = int(entry.attrib['team2'])
        d = int(entry.attrib['dist'])
        distances[i][j] = d

    L = U = None
    for c in root.findall('.//intp'):
        if c.get('mode1') == 'H':
            L, U = int(c.get('min')), int(c.get('max'))
            break
    if L is None:
        L, U = 1, 3

    constraints = {L,U}

    return n_teams, team_names, distances, constraints

n_teams, team_names, distances, constraints = parse_xml("Data/NL6.xml")
print("Number of teams:", n_teams)
print("Teams:", team_names)
print("Distances :")
for row in distances:
    print(row)


print("Constraints:", constraints)