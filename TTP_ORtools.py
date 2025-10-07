import xml.etree.ElementTree as ET
from ortools.sat.python import cp_model
import time


class TTPSolver:
    def __init__(self, filename):
        """Initialize solver with XML data"""
        self.filename = filename
        (self.n_teams,
         self.team_names,
         self.distances,
         self.L,
         self.U) = self.parse_ttp_data(filename)
        self.num_rounds = 2 * (self.n_teams - 1)
        print(f"Initialized TTP Solver:")
        print(f" Teams: {self.team_names}")
        print(f" Rounds: {self.num_rounds}, Capacity L={self.L}, U={self.U}")


    def parse_ttp_data(self, filename):
        """Parse XML file and extract tournament data"""
        tree = ET.parse(filename)
        root = tree.getroot()

        # Teams
        teams = sorted(root.findall('.//Teams/team'),
                       key=lambda t: int(t.attrib['id']))
        team_names = [t.attrib['name'] for t in teams]
        n = len(team_names)

        # Distances
        dist = [[0]*n for _ in range(n)]
        for e in root.findall('.//Distances/distance'):
            i = int(e.attrib['team1'])
            j = int(e.attrib['team2'])
            d = int(e.attrib['dist'])
            dist[i][j] = d

        # Capacity constraints L/U
        L = U = None
        for c in root.findall('.//intp'):
            if c.get('mode1') == 'H':
                L = int(c.get('min'))
                U = int(c.get('max'))
                break
        if L is None:
            L, U = 1, 3

        return n, team_names, dist, L, U

    def solve_exact(self, time_limit=300):
        """Build and solve CP-SAT model with y coupling constraints"""
        model = cp_model.CpModel()
        x, home, away = {}, {}, {}

        # x[i,j,r] - only create when i != j
        for i in range(self.n_teams):
            for j in range(self.n_teams):
                if i != j:
                    for r in range(self.num_rounds):
                        x[i,j,r] = model.NewBoolVar(f'x_{i}_{j}_{r}')

        # home/away flags
        for i in range(self.n_teams):
            for r in range(self.num_rounds):
                home[i,r] = model.NewBoolVar(f'home_{i}_{r}')
                away[i,r] = model.NewBoolVar(f'away_{i}_{r}')

        # y[i,s,t,r] - only create when s != t
        y = {}
        for i in range(self.n_teams):
            for s in range(self.n_teams):
                for t in range(self.n_teams):
                    if s != t:
                        for r in range(self.num_rounds-1):
                            y[i,s,t,r] = model.NewBoolVar(f'y_{i}_{s}_{t}_{r}')

        # 1) Double round robin
        for i in range(self.n_teams):
            for j in range(self.n_teams):
                if i != j:
                    model.Add(sum(x[i,j,r] for r in range(self.num_rounds)) == 1)
                    model.Add(sum(x[j,i,r] for r in range(self.num_rounds)) == 1)

        # 2) One game per round
        for i in range(self.n_teams):
            for r in range(self.num_rounds):
                games = [x[i,j,r] for j in range(self.n_teams) if i != j] + \
                        [x[j,i,r] for j in range(self.n_teams) if j != i]
                model.Add(sum(games) == 1)

        # 3) Home/Away definition
        for i in range(self.n_teams):
            for r in range(self.num_rounds):
                model.Add(home[i,r] == sum(x[j,i,r] for j in range(self.n_teams) if j != i))
                model.Add(away[i,r] == sum(x[i,j,r] for j in range(self.n_teams) if j != i))

        # 4) Capacity constraints L/U
        L, U = self.L, self.U
        for i in range(self.n_teams):
            # Max U consecutive
            for r in range(self.num_rounds - U):
                model.Add(sum(home[i,r+k] for k in range(U+1)) <= U)
                model.Add(sum(away[i,r+k] for k in range(U+1)) <= U)
            # Min L consecutive (simplified - may need more sophisticated logic)
            if L > 1:
                for r in range(self.num_rounds - L + 1):
                    # If starting a home streak, must continue for at least L rounds
                    if r > 0:
                        model.Add(home[i,r] - home[i,r-1] <= sum(home[i,r+k] for k in range(min(L, self.num_rounds-r))) / L)
                    # Similar for away streaks
                    if r > 0:
                        model.Add(away[i,r] - away[i,r-1] <= sum(away[i,r+k] for k in range(min(L, self.num_rounds-r))) / L)

        # 5) No repeaters (strengthened)
        for i in range(self.n_teams):
            for k in range(self.n_teams):
                if i != k:
                    for r in range(self.num_rounds-1):
                        model.Add(x[i,k,r] + x[k,i,r] + x[i,k,r+1] + x[k,i,r+1] <= 1)

        # 6) Travel coupling constraints (2.7-2.9) - FIXED
        for i in range(self.n_teams):
            for s in range(self.n_teams):
                for t in range(self.n_teams):
                    if s != t and i != s and i != t:  # Added guards for i != s and i != t
                        for r in range(self.num_rounds-1):
                            model.Add(y[i,s,t,r] >= x[i,s,r] + x[i,t,r+1] - 1)
                            model.Add(y[i,s,t,r] <= x[i,s,r])
                            model.Add(y[i,s,t,r] <= x[i,t,r+1])

        # Objective - minimize travel
        obj_terms = []
        
        # Travel between rounds
        for i in range(self.n_teams):
            for s in range(self.n_teams):
                for t in range(self.n_teams):
                    if s != t and i != s and i != t:  # Added guards
                        for r in range(self.num_rounds-1):
                            obj_terms.append(self.distances[s][t] * y[i,s,t,r])
        
        # Return home cost after last round
        for i in range(self.n_teams):
            for j in range(self.n_teams):
                if i != j:
                    obj_terms.append(self.distances[j][i] * x[i,j,self.num_rounds-1])

        if obj_terms:  # Only set objective if we have terms
            model.Minimize(sum(obj_terms))

        # Solve
        print("Solving...")
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit

        start = time.time()
        status = solver.Solve(model)
        t_solve = time.time() - start

        schedule = {}
        total_cost = None
        
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            schedule = self.extract_schedule(solver, x)
            total_cost = self.calculate_travel_cost(schedule)
            self.display_results(schedule, total_cost, t_solve, 
                               solver.StatusName(status), solver, home, away)
        else:
            print(f"No solution found. Status: {solver.StatusName(status)}")
            print(f"Solve time: {t_solve:.2f} seconds")

        return {
            'status': solver.StatusName(status),
            'time': t_solve,
            'cost': total_cost,
            'schedule': schedule
        }

    def calculate_travel_cost(self, schedule):
        """Calculate total travel cost for a schedule"""
        cost = 0
        loc = list(range(self.n_teams))
        for r in range(self.num_rounds):
            new_loc = loc.copy()
            for away, home in schedule[r]:
                cost += self.distances[loc[away]][home]
                new_loc[away] = home
            loc = new_loc
        # return home
        for i in range(self.n_teams):
            if loc[i] != i:
                cost += self.distances[loc[i]][i]
        return cost
    
    def calculate_team_travel_breakdown(self, schedule):
        """Per-team travel cost"""
        breakdown = {name:0 for name in self.team_names}
        loc = list(range(self.n_teams))
        for r in range(self.num_rounds):
            new_loc = loc.copy()
            for away, home in schedule[r]:
                d = self.distances[loc[away]][home]
                breakdown[self.team_names[away]] += d
                new_loc[away] = home
            loc = new_loc
        for i in range(self.n_teams):
            if loc[i] != i:
                breakdown[self.team_names[i]] += self.distances[loc[i]][i]
        return breakdown
    
    def extract_schedule(self, solver, x):
        """Extract schedule from CP-SAT variables"""
        sched = {}
        for r in range(self.num_rounds):
            sched[r] = []
            for i in range(self.n_teams):
                for j in range(self.n_teams):
                    if i!=j and solver.Value(x[i,j,r])==1:
                        sched[r].append((i,j))
        return sched
    
    def display_results(self, schedule, cost, time_s, status, solver, home, away):
        """Display full results"""
        print(f"\nSolver status: {status}")
        print(f"Solve time: {time_s:.2f}s, Total travel cost: {cost}\n")
        print("Schedule:")
        for r in range(self.num_rounds):
            print(f" Round {r+1}: ", end='')
            for a,h in schedule[r]:
                print(f"{self.team_names[a]}@{self.team_names[h]} ", end='')
            print()
        print("\nHome/Away pattern:")
        for i,name in enumerate(self.team_names):
            pattern = ''.join('H' if solver.Value(home[i,r])==1 else 'A'
                              for r in range(self.num_rounds))
            print(f" {name}: {pattern}")
        print("\nTravel breakdown:")
        for t,c in self.calculate_team_travel_breakdown(schedule).items():
            print(f" {t}: {c}")

    def print_xml_format(self, schedule):
        """Print schedule in XML format"""
        print("\n" + "="*50)
        print("XML FORMAT OUTPUT:")
        print("="*50)
        print('<?xml version="1.0" encoding="utf-8"?>')
        print('<Games>')
    
        for round_num in range(self.num_rounds):
            for away_team, home_team in schedule[round_num]:
              print(f'  <ScheduledMatch away="{away_team}" home="{home_team}" slot="{round_num}"/>')
    
        print('</Games>')

if __name__ == "__main__":
    solver = TTPSolver("Data/NL4.xml")
    result = solver.solve_exact(time_limit=300)

    if result['schedule']:
        solver.print_xml_format(result['schedule'])