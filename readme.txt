In this assignment we ask you to create a model (or more) to predict 1X2, i.e., probabilities of home team winning, draw taking place, or away team winning, for the remaining games of the season (any game after 2022-08-02).

In sportsbetting industry 1X2 is a bet offer, in which 1 denotes home team winning X denotes a draw, and 2 denotes away team winning. 
In event_name home team is listed first followed by the away team. 
The provided dataset consists of fixtures (time of games and results if applicable) and outrights (our odds for a team to win a season at the beginning of the season) for each football league (Liga Águila and Swedish Division 1 Södra).

Outrights:
    event_group - name of the league
    event_group_id - group identifiers
    event_dimension_id - match id
    event_name - name of the event group
    team
    odds - the first odds we have for the team to win the league/cup
    odds_year - the calendar year the odds were priced/set

Fixtures:
    event_dimension_id - match id
    date - date of the match
    closing_implied_prob_1 - implied probability of home team to win
    closing_implied_prob_X - implied probability of draw to take place
    closing_implied_prob_2 - implied probability of away team to win
    event_name - 'home team - away team'
    result - 'home goals:away goals'

Supportive questions:
- how to asses the model's accuracy?
- how to compare models performance (in case you create more than one)?
- what could go wrong?
