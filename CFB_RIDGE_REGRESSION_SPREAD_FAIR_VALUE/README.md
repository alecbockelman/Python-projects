# Python CFB Spread Calc
Uses the folowing stats trained on 2023 games:  
                'lines','head_to_head_favorites_win_pct','elo_rating_underdog_fav_diff','favorite_minus_underdog_stars',
                'start_day_of_week','fav_at_home_integer','underdog_at_home_integer','fav_offense_plays',
                'fav_offense_drives','fav_offense_ppa','fav_offense_total_ppa','fav_offense_success_rate',
                'fav_offense_explosiveness','fav_offense_power_success','fav_offense_stuff_rate','fav_offense_line_yards',
                'fav_offense_line_yards_total','fav_offense_second_level_yards','fav_offense_second_level_yards_total',
                'fav_offense_open_field_yards','fav_offense_open_field_yards_total',
                'fav_offense_total_opportunies','fav_offense_points_per_opportunity','fav_defense_plays',
                'fav_defense_drives','fav_defense_ppa','fav_defense_total_ppa','fav_defense_success_rate',
                'fav_defense_explosiveness','fav_defense_power_success','fav_defense_stuff_rate',
                'fav_defense_line_yards','fav_defense_line_yards_total','fav_defense_second_level_yards'
                ,'fav_defense_second_level_yards_total','fav_defense_open_field_yards',
                'fav_defense_open_field_yards_total','fav_defense_total_opportunies',
                'fav_defense_points_per_opportunity','fav_OCEI','fav_DCEI','underdog_offense_plays',
                'underdog_offense_drives','underdog_offense_ppa','underdog_offense_total_ppa',
                'underdog_offense_success_rate','underdog_offense_explosiveness','underdog_offense_power_success',
                'underdog_offense_stuff_rate','underdog_offense_line_yards','underdog_offense_line_yards_total',
                'underdog_offense_second_level_yards','underdog_offense_second_level_yards_total',
                'underdog_offense_open_field_yards','underdog_offense_open_field_yards_total','underdog_offense_total_opportunies',
                'underdog_offense_points_per_opportunity','underdog_defense_plays','underdog_defense_drives',
                'underdog_defense_ppa','underdog_defense_total_ppa','underdog_defense_success_rate',
                'underdog_defense_explosiveness','underdog_defense_power_success','underdog_defense_stuff_rate',
                'underdog_defense_line_yards','underdog_defense_line_yards_total','underdog_defense_second_level_yards',
                'underdog_defense_second_level_yards_total','underdog_defense_open_field_yards',
                'underdog_defense_open_field_yards_total','underdog_defense_total_opportunies',
                'underdog_defense_points_per_opportunity','underdog_OCEI','underdog_DCEI',
                'net_transfer_portal_favorite_minus_underdog_stars','fav_total_ppa','fav_total_passing_ppa',
                'fav_total_receiving_ppa','fav_total_rushing_ppa','fav_percent_ppa','fav_percent_passing_ppa',
                'fav_percent_receiving_ppa','fav_percent_rushing_ppa','fav_usage','fav_passing_usage',
                'fav_receiving_usage','fav_rushing_usage','underdog_total_ppa','underdog_total_passing_ppa',
                'underdog_total_receiving_ppa','underdog_total_rushing_ppa',
                'underdog_percent_ppa','underdog_percent_passing_ppa','underdog_percent_receiving_ppa',
                'underdog_percent_rushing_ppa','underdog_usage','underdog_passing_usage','underdog_receiving_usage',
                'underdog_rushing_usage'

                
1. Uses Ridge Regression to calculate fair spread, this regression model minimizes the loss value and has a penalty for an large coefficants to prevent Multicollinearity
   causing overfitting
2. Note code takes aprroximately **27.58 MINUTES TO RUN**
