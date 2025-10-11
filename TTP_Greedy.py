import random
import xml.etree.ElementTree as ET
import csv

class TTPGreedy:
    def __init__(self, filename):
        self.n, self.team_names, self.dist, self.L, self.U = self._parse_xml(filename)
        self.num_rounds = 2 * (self.n - 1)
        # Precompute all required games (double round-robin)
        self.all_games = [(i, j) for i in range(self.n) for j in range(self.n) if i != j]

    def _parse_xml(self, fn):
        tree = ET.parse(fn)
        root = tree.getroot()
        teams = sorted(root.findall('.//Teams/team'), key=lambda t: int(t.attrib['id']))
        names = [t.attrib['name'] for t in teams]
        n = len(names)
        dist = [[0]*n for _ in range(n)]
        for e in root.findall('.//Distances/distance'):
            i, j, d = int(e.attrib['team1']), int(e.attrib['team2']), int(e.attrib['dist'])
            dist[i][j] = d
        L = U = None
        for c in root.findall('.//intp'):
            if c.get('mode1') == 'H':
                L, U = int(c.get('min')), int(c.get('max'))
                break
        if L is None:
            L, U = 1, 3
        return n, names, dist, L, U

    def _is_valid(self, sched):
        # 1) exactly n/2 games per round and no team twice
        for r in sched:
            if len(r) != self.n//2: return False
            seen = set()
            for a,h in r:
                if a in seen or h in seen: return False
                seen.add(a); seen.add(h)
        # 2) each pair once
        cnt = {(i,j):0 for i,j in self.all_games}
        for r in sched:
            for g in r:
                cnt[g] += 1
        if any(v != 1 for v in cnt.values()): return False
        # 3) no-repeaters
        for k in range(self.num_rounds-1):
            curr = {tuple(sorted(g)) for g in sched[k]}
            nxt  = {tuple(sorted(g)) for g in sched[k+1]}
            if curr & nxt: return False
        # 4) capacity
        patterns = [[] for _ in range(self.n)]
        for r in sched:
            for a,h in r:
                patterns[a].append('A'); patterns[h].append('H')
        for pat in patterns:
            c = 1
            for i in range(1,len(pat)):
                if pat[i]==pat[i-1]:
                    c+=1
                    if c>self.U: return False
                else:
                    c=1
        return True

    def _travel_cost(self, sched):
        cost = 0
        loc = list(range(self.n))
        for r in range(self.num_rounds):
            newloc = loc.copy()
            for a,h in sched[r]:
                if loc[a] != h:
                    cost += self.dist[loc[a]][h]
                newloc[a] = h
            loc = newloc
        # return home
        for i in range(self.n):
            if loc[i] != i:
                cost += self.dist[loc[i]][i]
        return cost

    def create_schedule(self):
        """Greedy fill each round one by one"""
        sched = [[] for _ in range(self.num_rounds)]
        remaining = self.all_games.copy()
        random.shuffle(remaining)
        for r in range(self.num_rounds):
            teams_in_r = set()
            for game in remaining[:]:
                a,h = game
                if a not in teams_in_r and h not in teams_in_r and len(sched[r]) < self.n//2:
                    sched[r].append(game)
                    teams_in_r.update(game)
                    remaining.remove(game)
            # If incomplete or invalid, try to fill missing games
        # If incomplete or invalid, fallback round-robin
        if not self._is_valid(sched):
            return self._round_robin()
        return sched

    def _round_robin(self):
        """Standard round-robin construction"""
        teams = list(range(self.n))
        sched = [[] for _ in range(self.num_rounds)]
        for r in range(self.n-1):
            pairings = []
            for i in range(self.n//2):
                t1 = teams[i]; t2 = teams[-1-i]
                if r%2==0:
                    pairings.append((t1,t2))
                else:
                    pairings.append((t2,t1))
            sched[r] = pairings
            teams = [teams[0]] + [teams[-1]] + teams[1:-1]
        # mirror
        for r in range(self.n-1):
            sched[r+self.n-1] = [(h,a) for (a,h) in sched[r]]
        return sched

    def save_csv(self, sched, fname="greedy_schedule.csv"):
        with open(fname,"w",newline="") as f:
            writer=csv.writer(f)
            writer.writerow(["Round","Away_ID","Away_Name","Home_ID","Home_Name","Match"])
            for r in range(self.num_rounds):
                for a,h in sched[r]:
                    writer.writerow([r+1, a, self.team_names[a], h, self.team_names[h],
                                     f"{self.team_names[a]} @ {self.team_names[h]}"])
        print("Saved:", fname)

    def print_schedule(self, sched):
        """Print the schedule round by round with team names."""
        for r, round_games in enumerate(sched):
            print(f"Round {r+1}:")
            for away, home in round_games:
                print(f"  {self.team_names[away]} @ {self.team_names[home]}")
            print()

    def print_schedule_with_distances(self, sched):
        """
        Print the schedule with distances for each away trip.
        Also prints the cumulative distance per round and total.
        """
        total_distance = 0
        team_locations = list(range(self.n))  # start at home

        for r, round_games in enumerate(sched):
            print(f"Round {r+1}:")
            round_distance = 0
            new_locations = team_locations.copy()

            for away, home in round_games:
                from_city = team_locations[away]
                to_city = home
                dist = self.dist[from_city][to_city] if from_city != to_city else 0
                round_distance += dist
                total_distance += dist
                new_locations[away] = to_city

                print(f"  {self.team_names[away]} @ {self.team_names[home]}  "
                      f"(from {self.team_names[from_city]} to {self.team_names[to_city]}, dist={dist})")

            print(f"  Round {r+1} distance: {round_distance}\n")
            team_locations = new_locations

        # return home
        return_distance = 0
        for team in range(self.n):
            if team_locations[team] != team:
                dist = self.dist[team_locations[team]][team]
                return_distance += dist
                print(f"{self.team_names[team]} returns home from {self.team_names[team_locations[team]]}, dist={dist}")

        total_distance += return_distance
        print(f"\nReturn-home distance total: {return_distance}")
        print(f"Overall travel distance: {total_distance}")

if __name__=="__main__":
    tg = TTPGreedy("Data/NL6.xml")
    sched = tg.create_schedule()
    assert tg._is_valid(sched), "Greedy schedule invalid!"
    cost = tg._travel_cost(sched)
    print("Greedy cost:", cost)
    tg.print_schedule(sched)
    tg.print_schedule_with_distances(sched)


    #tg.save_csv(sched)
