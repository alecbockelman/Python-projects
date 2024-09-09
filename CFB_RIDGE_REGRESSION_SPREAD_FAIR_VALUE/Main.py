# -*- coding: utf-8 -*-
"""
Created by abockelman 9/08/2024
"""



import cfbd
from cfbd.rest import ApiException
import pandas as pd
import time
import numpy as np


''' Record Time for Program Speed Tests'''
start_time = time.time()


'''Establish API connection'''
configuration = cfbd.Configuration()
configuration.api_key['Authorization'] = '+9oTfEaNp0J+Pu8n0UYskeU0uZ19wVGtGZ1oUFY23RHHnXEj2kMW6mdJTq6oEtjt'
configuration.api_key_prefix['Authorization'] = 'Bearer'
api_instance = cfbd.BettingApi(cfbd.ApiClient(configuration))

'''Pull historical betting lines for all 2023 cfb games from Bovada'''
year=2023
season_type='regular'

try:
    '''Betting lines'''
    api_response = api_instance.get_lines(year=year, season_type=season_type)
except ApiException as e:
    print("Exception when calling BettingApi->get_lines: %s\n" % e)


'''convert response from json to dictionary and convert to pandas dataframe'''

api_response_pd = pd.DataFrame.from_records([item.to_dict() for item in api_response])
api_response_pd = api_response_pd.dropna()

'''Now extract Bovada Ml from dictionary nested in list and re-insert into pandas dataframe'''

api_response_pd =api_response_pd[api_response_pd['lines'].map(len) > 0]
api_response_pd.reset_index(drop=True,inplace=True)

count = 0
betting_lines_list = list()
favorite_list = list()
for i in api_response_pd['lines']:
    temp_list = list(filter(lambda d: d['provider']== 'Bovada',api_response_pd['lines'][count]))
    fav_list = list(filter(lambda d: d['provider']== 'Bovada',api_response_pd['lines'][count]))
    if len(temp_list) > 0:
        temp_list = [-1*abs(temp_list[0]['spread'])]
        spread_formatted = fav_list[0]['formatted_spread']
        remove_digits = ''.join([i for i in spread_formatted if not i.isdigit()])
        remove_digits = remove_digits.replace(' -.','')
        remove_digits = remove_digits.replace(' -','')
        
    else:
        temp_list = temp_list
        fav_list = fav_list
        
    betting_lines_list.append(temp_list)
    favorite_list.append(remove_digits)
    count +=1



'''Create Favorite Column'''

api_response_pd['lines']= betting_lines_list
api_response_pd['favorite'] = favorite_list
api_response_pd =api_response_pd[api_response_pd['lines'].map(len) > 0]
api_response_pd= api_response_pd.dropna()
api_response_pd.reset_index(drop=True,inplace=True)


    
formatted_lines_list = list()
# formatted_favorites_list = list()

for i in api_response_pd['lines']:
    formatted_lines_list.append(i[0])
    
# for i in api_response_pd['favorite']:
#     formatted_favorites_list.append(i[0])

api_response_pd['lines'] = formatted_lines_list
# api_response_pd['favorite'] = formatted_favorites_list



'''Calculate Winning team'''

count=0
winner_list=list()
for i in api_response_pd['lines']:
    if api_response_pd['home_score'][count] - api_response_pd['away_score'][count] > 0:
        winner_list.append(api_response_pd['home_team'][count])
    else:
        winner_list.append(api_response_pd['away_team'][count])
    count+=1
    
api_response_pd['winning_team']=winner_list


'''Calculate Actual Spread'''

api_response_pd['actual_spread'] = api_response_pd['home_score'] - api_response_pd['away_score']
api_response_pd['actual_spread'] = api_response_pd['actual_spread'].abs()*-1

'''Boolean Indexing'''
'''Change instance where favorite does not win to positive to reflect not covering and creating a continuous series'''
'''If favorite wins its negative if favorite loses its positive,
so if Texas A&M -15 and Texas A&M wins by 100 print -100 if they lose by 1pt print +1''
''This allows for the win/lose result to be captured as well as the spread into a single integer'''

api_response_pd.loc[api_response_pd['favorite'] != api_response_pd['winning_team'],'actual_spread']*=-1



'''Sortby game-id'''
api_response_pd.sort_values(by='id',ascending=True,inplace=True)
api_response_pd.reset_index(drop=True,inplace=True)



'''Add Factors for Gradient boosted decision Model'''


'''Establish teams   API connection'''
configuration = cfbd.Configuration()
configuration.api_key['Authorization'] = '+9oTfEaNp0J+Pu8n0UYskeU0uZ19wVGtGZ1oUFY23RHHnXEj2kMW6mdJTq6oEtjt'
configuration.api_key_prefix['Authorization'] = 'Bearer'
api_instance = cfbd.TeamsApi(cfbd.ApiClient(configuration))

'''Pull Team Matchup History'''

min_year = 2016
max_year = 2022


'''Iterate through df and insert favorites win pct for head to head over past 6 years'''

#create a function and only run once when training model

def head_to_head(min_year,max_year):
        list_fav_win_pct = list()
        favorite = str()
        count=0
        for i in api_response_pd['lines']:
            favorite = api_response_pd['favorite'][count]
            home = api_response_pd['home_team'][count]
            away = api_response_pd['away_team'][count]
            #determine which team is favorite, set that to team1 as home team
            if favorite == home:
                team1 = favorite
                team2 = away
            else:
                team1 = favorite
                team2 = home
                
            try:
                # Team matchup history
                api_response = api_instance.get_team_matchup(team1, team2, min_year=min_year, max_year=max_year)
            except ApiException as e:
                print("Exception when calling TeamsApi->get_team_matchup: %s\n" % e)
           
            
            win_pct_calc = pd.DataFrame.from_records(api_response.to_dict())
            win_pct_calc = win_pct_calc.dropna()
            if len(win_pct_calc) > 0:
                win_pct_calc = win_pct_calc['team1_wins'][0]/len(win_pct_calc)
            else:
                win_pct_calc = 0.5
            list_fav_win_pct.append(win_pct_calc)
            count+=1
        return list_fav_win_pct



'''Create new column for head to head stat'''

list_fav_win_pct = head_to_head(min_year, max_year)
api_response_pd['head_to_head_favorites_win_pct'] = list_fav_win_pct


'''Research indicates if favorite has winning record vs. opponent in the past 5 yrs they only cover spread 25% indicating spread is to aggressive
however if favorite has losing record vs opponent they cover spread about 50% means spread is roughly correctly priced'''

'''So for above any values not greater than 0.5 should be set to 0 for sharper signal and reduce noise when 
training model'''

#########
#########
####
####
api_response_pd.loc[api_response_pd['head_to_head_favorites_win_pct'] <= 0.5,'head_to_head_favorites_win_pct']=0
####
####
#########
#########

''''^^^^^^Uncomment line 180,181,190 to update the dataframe with head to head matchup ~ 12 minutes^^^^^^'''









'''Establish Ratings  API connection'''
configuration = cfbd.Configuration()
configuration.api_key['Authorization'] = '+9oTfEaNp0J+Pu8n0UYskeU0uZ19wVGtGZ1oUFY23RHHnXEj2kMW6mdJTq6oEtjt'
configuration.api_key_prefix['Authorization'] = 'Bearer'
api_instance = cfbd.RatingsApi(cfbd.ApiClient(configuration))




'''Iterate through df and calculate elo rating for home and away team for every week'''
'''Initial api pull outside loop but nested loop handles response'''




#create a function and only run once when training model

season_type = 'both'

def elo_rating(season_type):

        list_elo_diff_underdog_fav = list()
        game_ids = list()
        
        favorite = str()
        underdog = str()
        
        
        count=0
        for f in range(1,16):
            
            if f==1:
                week = 1
                year=2022
            if f > 1:
                week = f
                year = 2023
            else:
                week = f
                year = 2023
                
            #pull api data for favorite
            try:
                # Historical Elo ratings
                api_response = api_instance.get_elo_ratings(year=year, week=week, season_type=season_type)
                
            except ApiException as e:
                print("Exception when calling RatingsApi->get_elo_ratings: %s\n" % e)
           
            
            elo_rating = pd.DataFrame.from_records([item.to_dict() for item in api_response])
            elo_rating = elo_rating.dropna()

            
            sample_df = api_response_pd.loc[api_response_pd['week']==week]
            sample_df.reset_index(drop=True,inplace=True)
            
            count=0
            for i in sample_df['id']:
               favorite = sample_df['favorite'][count]
               home = sample_df['home_team'][count]
               away = sample_df['away_team'][count]

               #determine which team is underdog
               if favorite == home:
                   underdog = away
               else:
                   underdog = home   
                   
               elo_fav = elo_rating.loc[elo_rating['team']==favorite]['elo']
               elo_underdog = elo_rating.loc[elo_rating['team']==underdog]['elo']

               if elo_fav.empty or elo_underdog.empty:
                   elo_fav = 1500
                   elo_underdog =1500
               else:
                   elo_fav = elo_fav.values[0]
                   elo_underdog = elo_underdog.values[0]
               
                
               elo_rating_diff = elo_underdog - elo_fav
               list_elo_diff_underdog_fav.append(elo_rating_diff)
               
               game_ids.append(i)
               count+=1   
               
        return list_elo_diff_underdog_fav,game_ids        
            
           
            

            
            
            
          



'''Create new column for elo ratings stat'''

list_elo_diff_underdog_fav = elo_rating(season_type)[0]
game_ids = elo_rating(season_type)[1]

'''create a dicitonary then a sorted datframe on game-ids'''

data = {'id': game_ids, 'elo_rating_underdog_fav_diff':list_elo_diff_underdog_fav}
insert_df = pd.DataFrame(data)
insert_df.sort_values(by='id',ascending=True,inplace=True)


api_response_pd['elo_rating_underdog_fav_diff'] = insert_df['elo_rating_underdog_fav_diff']

  


'''Research indicates if underdog elo rating is greater than the favorite, 
then the favorite covers the spread 55% of the time,
and the favorite spread is mispriced most likely overcompensation for explosiveness'''


'''So for above... any values not greater than or equal to 1 should be set to 0
 for sharper signal and reduce noise when 
training model'''




#########
#########
####
####
api_response_pd.loc[api_response_pd['elo_rating_underdog_fav_diff'] < 0,'elo_rating_underdog_fav_diff']=0
####
####
#########
#########


'''^^^uncomment lines 305,306,310-312,315,330 to build the elo rating column^^^'''





start_year = 2021
end_year=2022

   

'''Establish Recruiting  API connection'''
configuration = cfbd.Configuration()
configuration.api_key['Authorization'] = '+9oTfEaNp0J+Pu8n0UYskeU0uZ19wVGtGZ1oUFY23RHHnXEj2kMW6mdJTq6oEtjt'
configuration.api_key_prefix['Authorization'] = 'Bearer'
api_instance = cfbd.RecruitingApi(cfbd.ApiClient(configuration))


'''Pulling only WR/QB positon data from 42 Analytics supports most important recruits'''


try:
    # Recruit position group ratings
    api_response = api_instance.get_recruiting_groups(start_year=start_year, end_year=end_year)
except ApiException as e:
    print("Exception when calling RecruitingApi->get_recruiting_groups: %s\n" % e)

recruit_groups_team = pd.DataFrame.from_records([item.to_dict() for item in api_response])
recruit_groups_team = recruit_groups_team.dropna()

recruit_groups_team = recruit_groups_team.loc[recruit_groups_team['position_group'].isin(['Quarterback','Receiver','Defensive Back'])]

recruit_groups_team =recruit_groups_team.groupby('team',group_keys=False)[['total_rating']].apply(lambda x: x.mean())



'''Create new column for underdog_stars and fav_stars'''

favorite = str()
underdog = str()
list_underdog=list()

list_underdog_stars = list()
list_fav_stars = list()

list_fav_minus_underdog_stars = list()

count=0
for i in api_response_pd['lines']:
    favorite = api_response_pd['favorite'][count]
    home = api_response_pd['home_team'][count]
    away = api_response_pd['away_team'][count]
    #determine which team is favorite, set that to team1 as home team
    if favorite == home:
        underdog = away
    else:
        underdog=home

    if len(recruit_groups_team[recruit_groups_team.index==underdog]) > 0 and len(recruit_groups_team[recruit_groups_team.index==favorite]) > 0:
        selected_underdog = recruit_groups_team[recruit_groups_team.index==underdog].values[0][0]
        selected_fav = recruit_groups_team[recruit_groups_team.index==favorite].values[0][0]
        
        list_underdog_stars.append(selected_underdog)
        list_fav_stars.append(selected_fav)
        
    else:
        list_underdog_stars.append(0)
        list_fav_stars.append(0)
    list_underdog.append(underdog)    
    count+=1

list_fav_minus_underdog_stars = np.subtract(list_fav_stars,list_underdog_stars).tolist()

'''Add new Column for difference of favorite recruit position rank vs underdog recruit position rank'''

api_response_pd['favorite_minus_underdog_stars'] = list_fav_minus_underdog_stars


'''Create Underdog Column'''

api_response_pd['underdog'] = list_underdog
api_response_pd= api_response_pd.dropna()
api_response_pd.reset_index(drop=True,inplace=True)







'''no need to adjust.. note that any teams that have no recruits have a 0 score'''

'''Research suggest that when the underdog has better qb/lb then the favorite only covers 46% of the time'''
'''Research suggest that when the underdog has better qb/wr then the favorite only covers 43% of the time'''



###Stopped here, need to pull in stats should I use returning production, its only ppa??
###should test out both



'''Create feature for weekday of game as a variable 1 = sunday and 7= saturday'''
api_response_pd['start_day_of_week'] = pd.to_datetime(api_response_pd['start_date']).dt.weekday

'''Research shows games on Thur or Friday favorite has a 57% of covering
   games on sunday the favorite chance to cover drops to 45%'''


'''Set non Sunday,Thur,or Fri games to 0 for day of week variable'''

api_response_pd.loc[~api_response_pd['start_day_of_week'].isin([6,4,3]),'start_day_of_week']=0








'''Research shows only a few teams have actually stats sig home feilds here are the ones selected:
    Michigan,Penn State, Ohio State, Texas A&M, Florida, LSU, USC, Texas,Clemson,etc'''


'''When one of these teams is favorite and at home they have a 50% chance of winning,
when underdogs they  at home sample HAVE A 62% OF COVERING SPREAD'''


'''Manual create column for super location integer home/away data, if home =1, else =0'''

super_locations =['Texas A&M','LSU','Ohio State','Georgia', 'Penn State', 'Wisconsin',
                  'Oklahoma', 'Florida State','Florida', 'Oregon', 'Clemson', 'Tennessee',
                  'Auburn', 'South Carolina', 'Michigan', 'USC']


#create zero filled columns
api_response_pd['fav_at_home_integer'] =0
api_response_pd['underdog_at_home_integer'] = 0


#assign 1 if fav home or underdog at home
api_response_pd.loc[api_response_pd['favorite'] == api_response_pd['home_team'],'fav_at_home_integer']=1
api_response_pd.loc[api_response_pd['favorite'] != api_response_pd['home_team'],'underdog_at_home_integer']=1

#change the 1 to 0 if team is not a super_location
api_response_pd.loc[~api_response_pd['favorite'].isin(super_locations),'fav_at_home_integer']=0
api_response_pd.loc[~api_response_pd['favorite'].isin(super_locations),'underdog_at_home_integer']=0





#########
#########
####
####
api_response_pd.loc[api_response_pd['elo_rating_underdog_fav_diff'] < 0,'elo_rating_underdog_fav_diff']=0
####
####
#########
#########






'''Establish stats API connection'''
configuration = cfbd.Configuration()
configuration.api_key['Authorization'] = '+9oTfEaNp0J+Pu8n0UYskeU0uZ19wVGtGZ1oUFY23RHHnXEj2kMW6mdJTq6oEtjt'
configuration.api_key_prefix['Authorization'] = 'Bearer'
api_instance = cfbd.StatsApi(cfbd.ApiClient(configuration))


'''Pull historical all possible stats for every team for the 2022 cfb season, use for start of next season'''
year=2022
exclude_garbage_time = True

try:
    '''stats'''
    api_response = api_instance.get_advanced_team_season_stats(year=year, exclude_garbage_time=exclude_garbage_time)
except ApiException as e:
    print("Exception when calling StatsApi->get_advanced_team_season_stats: %s\n" % e)



'''convert response from json to dictionary and convert to pandas dataframe'''

api_stats = pd.DataFrame.from_records([item.to_dict() for item in api_response])
api_stats = api_stats.dropna()




'''Filter Team stats on unique Teams list'''

unique_home_team_names = api_response_pd['home_team'].unique().tolist()
unique_away_team_names = api_response_pd['away_team'].unique().tolist()
total_team_names = unique_home_team_names+unique_away_team_names

api_stats = api_stats[api_stats['team'].isin(total_team_names)]




'''uncompress dictionary that is nested in pandas dataframe
    create individual stats columns for offense and defense'''
    
    
'''layer1 and layer2 stats layer 1 are used as is, layer2 requires expanding dictionary'''
    
stats_layer1 = ['plays', 'drives', 'ppa', 'total_ppa',
'success_rate', 'explosiveness', 'power_success', 'stuff_rate', 
'line_yards', 'line_yards_total', 'second_level_yards', 
'second_level_yards_total','open_field_yards', 'open_field_yards_total', 
'total_opportunies', 'points_per_opportunity']

#    
'''new calc called chaotic effiecentcy index'''
#offense = havoc_frontseven +standarddowns success rate+(passing downs explos+passsing plays explo)*passing plays success rate
# + (rushing down explo+ppa_rushing_plays)*rushing plays success rate

#defense = havoc_frontseven +standarddowns success rate+(passing downs explos+passsing plays explo)*passing plays success rate
# + (rushing down explo+ppa_rushing_plays)*rushing plays success rate
 
    


'''Offense & defense stats all of prior season'''
'''Need to repeat with retutning players for sharper numbers'''

api_stats.reset_index(drop=True,inplace=True)

df_ex_database = []
count = 0     
for k in api_stats['team']:
    ##see if there is a way to append data to pd.df line by line to 'ex_dataset' I think I can do this using dictionary and.loc
    
    layer1_row_as_dict_type = {'team': k}
    
    #offense
    for n in stats_layer1:
      new_str = 'offense_'+n
      #print(new_str,n)
      layer1_row_as_dict_type[new_str] =api_stats['offense'][count][n]
    
    #defense
    for n in stats_layer1:
      new_str = 'defense_'+n
      layer1_row_as_dict_type[new_str] =api_stats['defense'][count][n]
    
     #offense chaotic eff index(OCEI) 
    ocei = api_stats['offense'][count]['havoc']['front_seven'] 
    ocei += api_stats['offense'][count]['standard_downs']['success_rate']
    ocei += ((api_stats['offense'][count]['passing_downs']['explosiveness']+api_stats['offense'][count]['passing_plays']['explosiveness'])*api_stats['offense'][count]['passing_plays']['success_rate'])
    ocei +=  ((api_stats['offense'][count]['standard_downs']['explosiveness']+api_stats['offense'][count]['rushing_plays']['ppa'])*api_stats['offense'][count]['rushing_plays']['success_rate'])
    
    #offense chaotic eff index(DCEI) 
    dcei = api_stats['defense'][count]['havoc']['front_seven'] 
    dcei -= api_stats['defense'][count]['standard_downs']['success_rate']
    dcei -= ((api_stats['defense'][count]['passing_downs']['explosiveness']+api_stats['defense'][count]['passing_plays']['explosiveness'])*api_stats['defense'][count]['passing_plays']['success_rate'])
    dcei -=  ((api_stats['defense'][count]['standard_downs']['explosiveness']+api_stats['defense'][count]['rushing_plays']['ppa'])*api_stats['defense'][count]['rushing_plays']['success_rate'])
   
    layer1_row_as_dict_type['OCEI'] = ocei
    layer1_row_as_dict_type['DCEI'] = dcei   
    
    df_ex_database.append(layer1_row_as_dict_type)  
    count+=1
    

    
    
df_ex_database = pd.DataFrame.from_dict(df_ex_database, orient='columns')



#merge ex_dataset and api_response_pd on home_team name and team name


##then rename all column stats to home_off_stats
##then repeat merge and rename with away_stats

###create OCEI and DCEI difference stat, leave reminder as outright vals


'''Merge ex_dataset to main dataframe on the key 'favorite' column
rename column for merge operation
rename all columns to fav_stat_name
repeat above for underdog and then create a diff column
'''
df_ex_database.rename(columns={'team':'favorite'},inplace = True)
merged_api = pd.merge(api_response_pd,df_ex_database, on='favorite')

merged_api.rename(columns={'offense_plays':'fav_offense_plays', 'offense_drives':'fav_offense_drives', 
                           'offense_ppa':'fav_offense_ppa', 'offense_total_ppa':'fav_offense_total_ppa',
                           'offense_success_rate': 'fav_offense_success_rate','offense_success_rate':'fav_offense_success_rate',
                           'offense_explosiveness': 'fav_offense_explosiveness','offense_power_success':'fav_offense_power_success',
                           'offense_stuff_rate':'fav_offense_stuff_rate', 'offense_line_yards':'fav_offense_line_yards',
                           'offense_line_yards_total':'fav_offense_line_yards_total','offense_second_level_yards':'fav_offense_second_level_yards',
                           'offense_second_level_yards_total':'fav_offense_second_level_yards_total', 
                           'offense_open_field_yards':'fav_offense_open_field_yards','offense_open_field_yards_total':'fav_offense_open_field_yards_total',
                           'offense_total_opportunies':'fav_offense_total_opportunies','offense_points_per_opportunity':'fav_offense_points_per_opportunity',
                           'defense_plays': 'fav_defense_plays','defense_drives': 'fav_defense_drives','defense_ppa': 'fav_defense_ppa',
                           'defense_total_ppa':'fav_defense_total_ppa', 'defense_success_rate':'fav_defense_success_rate',
                           'defense_explosiveness':'fav_defense_explosiveness','defense_power_success': 'fav_defense_power_success', 
                           'defense_stuff_rate':'fav_defense_stuff_rate','defense_line_yards':'fav_defense_line_yards',
                           'defense_line_yards_total':'fav_defense_line_yards_total', 'defense_second_level_yards':'fav_defense_second_level_yards',
                           'defense_second_level_yards_total': 'fav_defense_second_level_yards_total','defense_open_field_yards':'fav_defense_open_field_yards',
                           'defense_open_field_yards_total':'fav_defense_open_field_yards_total','defense_total_opportunies':'fav_defense_total_opportunies',
                           'defense_points_per_opportunity':'fav_defense_points_per_opportunity', 'OCEI':'fav_OCEI','DCEI':'fav_DCEI'},inplace = True)


merged_api= merged_api.fillna(0)
merged_api.reset_index(drop=True,inplace=True)






df_ex_database.rename(columns={'favorite':'underdog'},inplace = True)
merged_api = pd.merge(merged_api,df_ex_database, on='underdog')

merged_api.rename(columns={'offense_plays':'underdog_offense_plays', 'offense_drives':'underdog_offense_drives', 
                           'offense_ppa':'underdog_offense_ppa', 'offense_total_ppa':'underdog_offense_total_ppa',
                           'offense_success_rate': 'underdog_offense_success_rate','offense_success_rate':'underdog_offense_success_rate',
                           'offense_explosiveness': 'underdog_offense_explosiveness','offense_power_success':'underdog_offense_power_success',
                           'offense_stuff_rate':'underdog_offense_stuff_rate', 'offense_line_yards':'underdog_offense_line_yards',
                           'offense_line_yards_total':'underdog_offense_line_yards_total','offense_second_level_yards':'underdog_offense_second_level_yards',
                           'offense_second_level_yards_total':'underdog_offense_second_level_yards_total', 
                           'offense_open_field_yards':'underdog_offense_open_field_yards','offense_open_field_yards_total':'underdog_offense_open_field_yards_total',
                           'offense_total_opportunies':'underdog_offense_total_opportunies','offense_points_per_opportunity':'underdog_offense_points_per_opportunity',
                           'defense_plays': 'underdog_defense_plays','defense_drives': 'underdog_defense_drives','defense_ppa': 'underdog_defense_ppa',
                           'defense_total_ppa':'underdog_defense_total_ppa', 'defense_success_rate':'underdog_defense_success_rate',
                           'defense_explosiveness':'underdog_defense_explosiveness','defense_power_success': 'underdog_defense_power_success', 
                           'defense_stuff_rate':'underdog_defense_stuff_rate','defense_line_yards':'underdog_defense_line_yards',
                           'defense_line_yards_total':'underdog_defense_line_yards_total', 'defense_second_level_yards':'underdog_defense_second_level_yards',
                           'defense_second_level_yards_total': 'underdog_defense_second_level_yards_total','defense_open_field_yards':'underdog_defense_open_field_yards',
                           'defense_open_field_yards_total':'underdog_defense_open_field_yards_total','defense_total_opportunies':'underdog_defense_total_opportunies',
                           'defense_points_per_opportunity':'underdog_defense_points_per_opportunity', 'OCEI':'underdog_OCEI','DCEI':'underdog_DCEI'},inplace = True)


merged_api= merged_api.fillna(0)
merged_api.reset_index(drop=True,inplace=True)


"""ATTENTION: API_RESPONSE_PD IS NOW MERGED_API """



'''Transfer Portal Stats'''

year = 2023 ###This was 2022 before?? will stats drop??

   

'''Establish Transfer Portal  API connection'''
configuration = cfbd.Configuration()
configuration.api_key['Authorization'] = '+9oTfEaNp0J+Pu8n0UYskeU0uZ19wVGtGZ1oUFY23RHHnXEj2kMW6mdJTq6oEtjt'
configuration.api_key_prefix['Authorization'] = 'Bearer'
api_instance = cfbd.PlayersApi(cfbd.ApiClient(configuration))


'''Pulling all positions on hypothesis that transfer players are likely to start immediately'''


try:
    # Transfer portal by season
    api_response = api_instance.get_transfer_portal(year)
except ApiException as e:
    print("Exception when calling PlayersApi->get_transfer_portal: %s\n" % e)

transfer_portal_team = pd.DataFrame.from_records([item.to_dict() for item in api_response])
transfer_portal_team= transfer_portal_team.dropna()

'''Only selected transfers that have immediate eligibility'''

transfer_portal_team= transfer_portal_team.loc[transfer_portal_team['eligibility']=='Immediate']


'''Might not have a position effect like recruiting'''
#transfer_portal_team= transfer_portal_team.loc[transfer_portal_team['position'].isin(['QB','WR','CB','S'])]

transfer_portal_team_new =transfer_portal_team.groupby('destination',group_keys=False)[['stars']].apply(lambda x: x.sum())
transfer_portal_team_exit = transfer_portal_team.groupby('origin',group_keys=False)[['stars']].apply(lambda x: x.sum())

transfer_portal_team_net = transfer_portal_team_new.sub(transfer_portal_team_exit, fill_value=0).fillna(0)

'''Create new column for underdog_transfer and fav_transfer'''

favorite = str()
underdog = str()

list_underdog_transfer = list()
list_fav_transfer = list()

list_fav_minus_underdog_transfer = list()

count=0
for i in merged_api['lines']:
    favorite = merged_api['favorite'][count]
    home = merged_api['home_team'][count]
    away = merged_api['away_team'][count]
    #determine which team is favorite, set that to team1 as home team
    if favorite == home:
        underdog = away
    else:
        underdog=home

    if len(transfer_portal_team_net[transfer_portal_team_net.index==underdog]) > 0 and len(transfer_portal_team_net[transfer_portal_team_net.index==favorite]) > 0:
        selected_underdog = transfer_portal_team_net[transfer_portal_team_net.index==underdog].values[0][0]
        selected_fav = transfer_portal_team_net[transfer_portal_team_net.index==favorite].values[0][0]
        
        list_underdog_transfer.append(selected_underdog)
        list_fav_transfer.append(selected_fav)
        
    else:
        list_underdog_transfer.append(0)
        list_fav_transfer.append(0)
    list_underdog.append(underdog)    
    count+=1

list_fav_minus_underdog_transfer = np.subtract(list_fav_transfer,list_underdog_transfer).tolist()

'''Add new Column for difference of favorite net transfer portal - underdog net transfer portal'''



merged_api['net_transfer_portal_favorite_minus_underdog_stars'] = list_fav_minus_underdog_transfer

merged_api= merged_api.fillna(0)
merged_api.reset_index(drop=True,inplace=True)







'''Now Pull all Returning player stats'''


list_returning_production_diff = list()
favorite = str()
underdog = str()


#pull api data for favorite
try:
    # Team returning production metrics
    api_response = api_instance.get_returning_production(year=year)
except ApiException as e:
    print("Exception when calling PlayersApi->get_returning_production: %s\n" % e)
           

returning_stats = pd.DataFrame.from_records([item.to_dict() for item in api_response])
returning_stats = returning_stats[returning_stats['team'].isin(total_team_names)]    



'''Stats Layer1 there are no additional stats'''

stats_layer1 = ['total_ppa', 'total_passing_ppa',
       'total_receiving_ppa', 'total_rushing_ppa', 'percent_ppa',
       'percent_passing_ppa', 'percent_receiving_ppa', 'percent_rushing_ppa',
       'usage', 'passing_usage', 'receiving_usage', 'rushing_usage']


'''Offense & defense stats all of prior season'''
'''Need to repeat with retutning players for sharper numbers'''

returning_stats.reset_index(drop=True,inplace=True)



#########
#########Break, above is clean code
#########




'''Merge returning_stats to main dataframe on the key 'favorite' column
rename column for merge operation
rename all columns to fav_stat_name
repeat above for underdog
'''
returning_stats.rename(columns={'team':'favorite'},inplace = True)
merged_api = pd.merge(merged_api,returning_stats,on='favorite')

merged_api.rename(columns={'total_ppa':'fav_total_ppa', 'total_passing_ppa':'fav_total_passing_ppa',
       'total_receiving_ppa':'fav_total_receiving_ppa', 'total_rushing_ppa':'fav_total_rushing_ppa', 
       'percent_ppa':'fav_percent_ppa','percent_passing_ppa':'fav_percent_passing_ppa', 'percent_receiving_ppa':'fav_percent_receiving_ppa',
       'percent_rushing_ppa':'fav_percent_rushing_ppa','usage':'fav_usage', 'passing_usage':'fav_passing_usage',
       'receiving_usage':'fav_receiving_usage', 'rushing_usage':'fav_rushing_usage'},inplace = True)


merged_api =merged_api.fillna(0)
merged_api.reset_index(drop=True,inplace=True)




returning_stats.rename(columns={'favorite':'underdog'},inplace = True)
merged_api = pd.merge(merged_api,returning_stats, on = 'underdog')

merged_api.rename(columns={'total_ppa':'underdog_total_ppa', 'total_passing_ppa':'underdog_total_passing_ppa',
       'total_receiving_ppa':'underdog_total_receiving_ppa', 'total_rushing_ppa':'underdog_total_rushing_ppa', 
       'percent_ppa':'underdog_percent_ppa','percent_passing_ppa':'underdog_percent_passing_ppa', 'percent_receiving_ppa':'underdog_percent_receiving_ppa',
       'percent_rushing_ppa':'underdog_percent_rushing_ppa','usage':'underdog_usage', 'passing_usage':'underdog_passing_usage',
       'receiving_usage':'underdog_receiving_usage', 'rushing_usage':'underdog_rushing_usage'},inplace = True)


merged_api =merged_api.fillna(0)
merged_api.reset_index(drop=True,inplace=True)




'''RENAME MERGED_API TO FINAL_DATA'''
final_data = merged_api

'''Create target column that equals diff between betting line-actual'''
final_data['target'] = final_data['lines']-final_data['actual_spread']



'''Format Data'''
final_data = final_data.loc[final_data['lines']>=-16]
final_data = final_data.fillna(0)
final_data.reset_index(drop=True,inplace=True)





####DROP ANY SPREADS LESS THAN -16(NOT GOING TO PREDICT GAMES EXPECTED TO BE 3+ SCORE BLOWOUTS, PLUS THIS MAY SKEW RESULTS)


"""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""
'''BUILD RIDGE REGRESSION MODEL'''
"""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""


#matplotlib inline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import scale 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

'''We will use the sklearn package in order to perform ridge regression and the lasso.
 The main functions in this package that we care about are Ridge(), which can be 
 used to fit ridge regression models, and Lasso() which will fit lasso models. 
 They also have cross-validated counterparts: RidgeCV() and LassoCV(). We'll use these a bit later.'''


#assign target column to y variable

y = final_data.target
#y = final_data.actual_spread

#Here are all the features we will use, any calculated variables showed noticeable significance in predicitng spread
#and should be include

#remove non numeric variables, also remove betting line as that would most likely result in overfitting
#below is the feature set we will use


X = final_data[['lines','head_to_head_favorites_win_pct','elo_rating_underdog_fav_diff','favorite_minus_underdog_stars',
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
                'underdog_rushing_usage']]


X.info()


'''The Ridge() function has an alpha argument ( λ) that is used to tune the model. 
 We'll generate an array of alpha values ranging from very big to very small,
 essentially covering the full range of scenarios from the null model
 containing only the intercept, to the least squares fit'''
 
 

alphas = 2**np.linspace(10,-2,100)*0.5
alphas


''''Associated with each alpha value is a vector of ridge regression coefficients, 
  which we'll store in a matrix coefs. In this case, it is a  100×100
  matrix, with 100 rows (one for each predictor) and 100 columns (one for each value of alpha). 
  Remember that we'll want to standardize the variables so that they are on the same scale. 
  To do this, we can use the normalize = True parameter:'''
      

ridge = Ridge(solver='auto')

'''The precision of the solution (coef_) is determined by 
tol which specifies a different convergence criterion for each solver:
    ‘sparse_cg’: norm of residuals smaller than tol.
  
   using ‘sparse_cg’ as my solver
   this solver uses the conjugate gradient solver as found in scipy.sparse.linalg.cg. 
   As an iterative algorithm, this solver is more appropriate than ‘cholesky’ for large-scale data 
   (possibility to set tol and max_iter).
   '''

###VERY IMPORTANT TO NORMALIZE MY VARIABLES BEFORE USING A RIDGE REGRESSION,
###THIS WILL MAKE SURE ALL VALUES ARE ON THE SAME SCALE
###

# Create a StandardScaler object
scaler = StandardScaler()

# Fit and transform the data
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

coefs = []

for a in alphas:
    ridge.set_params(alpha = a)
    ridge.fit(X, y)
    coefs.append(ridge.coef_)
    
np.shape(coefs)

'''We expect the coefficient estimates to be much smaller, in terms of  l2
  norm, when a large value of alpha is used, as compared to when a small value of alpha is used. Let's plot and find out:'''

ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')


# Split data into training and test sets
X_train, X_test , y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)


'''Use ridge regression model and print MSE on the test set, using  λ=4'''

ridge2 = Ridge(alpha = 4, solver='auto')
ridge2.fit(X_train, y_train)             # Fit a ridge regression on the training data
pred2 = ridge2.predict(X_test)           # Use this model to predict the test data
print(pd.Series(ridge2.coef_, index = X.columns)) # Print coefficients
print(mean_squared_error(y_test, pred2))          # Calculate the test MSE


'''test with Decently large alpha of alpha = 2^5 '''
ridge3 = Ridge(alpha = 2**5, solver='auto')
ridge3.fit(X_train, y_train)             # Fit a ridge regression on the training data
pred3 = ridge3.predict(X_test)           # Use this model to predict the test data
print(pd.Series(ridge3.coef_, index = X.columns)) # Print coefficients
print(mean_squared_error(y_test, pred3))          # Calculate the test MSE



'''test with extremly large alpha of alpha = 10^100, should see MSE lower but will force coef to be almost 0 '''
ridge4 = Ridge(alpha = 10**10, solver='auto')
ridge4.fit(X_train, y_train)             # Fit a ridge regression on the training data
pred4 = ridge4.predict(X_test)           # Use this model to predict the test data
print(pd.Series(ridge4.coef_, index = X.columns)) # Print coefficients
print(mean_squared_error(y_test, pred4))          # Calculate the test MSE



'''Compare alpha of 4, 10^10, and 0. At alpha =0 will just be least squares simple regression,
   so in order to a valid and relevant model our MSE must be lower than the MSE at alpha =0'''

ridge0 = Ridge(alpha = 0, solver='auto')
ridge0.fit(X_train, y_train)             # Fit a ridge regression on the training data
pred = ridge0.predict(X_test)            # Use this model to predict the test data
print(pd.Series(ridge0.coef_, index = X.columns)) # Print coefficients
print(mean_squared_error(y_test, pred))           # Calculate the test MSE



#####GOOD NEWS! Our model beats a simple regression model of alpha =0, the 
#####optimal alpha lies somewhere between 4 and 512, anything higher will 
##### reduce coef to be almost zero


###Find optimal alpha that minimizes the cross validated ridge regression function

#Force eigenvalue for covar of matrix instead of default of auto

ridgecv = RidgeCV(alphas = alphas, scoring = 'neg_mean_squared_error', alpha_per_target = False, gcv_mode='eigen')
ridgecv.fit(X_train, y_train)
ridgecv.alpha_

ridge5 = Ridge(alpha = 2**5, solver='auto')
ridge5.fit(X_train, y_train)
mean_squared_error(y_test, ridge5.predict(X_test))

'''Need to look at different solver methods other than sparse_cg, it seems to max out alpha, I even had to cap alphas at 512 and
512 was selected'''

ridge5.fit(X, y)

resulting_coefs= pd.Series(ridge5.coef_, index = X.columns)
resulting_coefs 




'''Now backtest with output for a expected_value vs observed_value'''

'''Also this model will not include y-int it does not make sense to have any directional bias when predciting spread,
    and to clarify we are not actually predicting the spread, we are predicting between the actual spread and betting line from Bovada,
    so even more reason for y-int = 0.'''
    
    
'''Iterate through 2023 in sample, then check out of sample 2023 for lines other than bovada'''
'''Also check if other years have similar win rate if not may need to retrain model'''


'''When Backtesting try total wins against the spread no normalizing and then also try scoring cover vs. no cover of top 4 vs bottom 4
with normalizing and grabbing highest confidence favs and upsets'''




'''SEASON 2023 INTIAL SIMPLE BACKTEST'''
#pulled features + X(which is normalized) to append prediction column to final_data
backtest = final_data

model_features = ['lines', 'head_to_head_favorites_win_pct',
       'elo_rating_underdog_fav_diff', 'favorite_minus_underdog_stars',
       'start_day_of_week', 'fav_at_home_integer', 'underdog_at_home_integer',
       'fav_offense_plays', 'fav_offense_drives', 'fav_offense_ppa',
       'fav_offense_total_ppa', 'fav_offense_success_rate',
       'fav_offense_explosiveness', 'fav_offense_power_success',
       'fav_offense_stuff_rate', 'fav_offense_line_yards',
       'fav_offense_line_yards_total', 'fav_offense_second_level_yards',
       'fav_offense_second_level_yards_total', 'fav_offense_open_field_yards',
       'fav_offense_open_field_yards_total', 'fav_offense_total_opportunies',
       'fav_offense_points_per_opportunity', 'fav_defense_plays',
       'fav_defense_drives', 'fav_defense_ppa', 'fav_defense_total_ppa',
       'fav_defense_success_rate', 'fav_defense_explosiveness',
       'fav_defense_power_success', 'fav_defense_stuff_rate',
       'fav_defense_line_yards', 'fav_defense_line_yards_total',
       'fav_defense_second_level_yards',
       'fav_defense_second_level_yards_total', 'fav_defense_open_field_yards',
       'fav_defense_open_field_yards_total', 'fav_defense_total_opportunies',
       'fav_defense_points_per_opportunity', 'fav_OCEI', 'fav_DCEI',
       'underdog_offense_plays', 'underdog_offense_drives',
       'underdog_offense_ppa', 'underdog_offense_total_ppa',
       'underdog_offense_success_rate', 'underdog_offense_explosiveness',
       'underdog_offense_power_success', 'underdog_offense_stuff_rate',
       'underdog_offense_line_yards', 'underdog_offense_line_yards_total',
       'underdog_offense_second_level_yards',
       'underdog_offense_second_level_yards_total',
       'underdog_offense_open_field_yards',
       'underdog_offense_open_field_yards_total',
       'underdog_offense_total_opportunies',
       'underdog_offense_points_per_opportunity', 'underdog_defense_plays',
       'underdog_defense_drives', 'underdog_defense_ppa',
       'underdog_defense_total_ppa', 'underdog_defense_success_rate',
       'underdog_defense_explosiveness', 'underdog_defense_power_success',
       'underdog_defense_stuff_rate', 'underdog_defense_line_yards',
       'underdog_defense_line_yards_total',
       'underdog_defense_second_level_yards',
       'underdog_defense_second_level_yards_total',
       'underdog_defense_open_field_yards',
       'underdog_defense_open_field_yards_total',
       'underdog_defense_total_opportunies',
       'underdog_defense_points_per_opportunity', 'underdog_OCEI',
       'underdog_DCEI', 'net_transfer_portal_favorite_minus_underdog_stars',
       'fav_total_ppa', 'fav_total_passing_ppa', 'fav_total_receiving_ppa',
       'fav_total_rushing_ppa', 'fav_percent_ppa', 'fav_percent_passing_ppa',
       'fav_percent_receiving_ppa', 'fav_percent_rushing_ppa', 'fav_usage',
       'fav_passing_usage', 'fav_receiving_usage', 'fav_rushing_usage',
       'underdog_total_ppa', 'underdog_total_passing_ppa',
       'underdog_total_receiving_ppa', 'underdog_total_rushing_ppa',
       'underdog_percent_ppa', 'underdog_percent_passing_ppa',
       'underdog_percent_receiving_ppa', 'underdog_percent_rushing_ppa',
       'underdog_usage', 'underdog_passing_usage', 'underdog_receiving_usage',
       'underdog_rushing_usage']


#Create expected_value column using sumproduct
backtest['expected_value'] = (X[model_features]*resulting_coefs).sum(axis=1)
#backtest['expected_spread'] = (X[model_features]*resulting_coefs).sum(axis=1)

#Now add expected value column to lines to get projected_spread.THEN MULTIPLY BY -1 TO GET CORRECT DIRECTION, NEED TO PAST FORMULA BELOW:
    #target = lines -actual
    #expected value = lines - actual
    #expected_value -lines = -actual
    #-expected_value +lines = actual
backtest['projected_spread'] = -1*backtest['expected_value'] +backtest['lines']



""""""""""""""
""""""""""""""""""
""""""""""""""""""
""""""""""""""""""
""""""""""""""""""
""""""""""""""""""
""""""""""""""""""
'''BACKTEST'''
""""""""""""""
""""""""""""""""""
""""""""""""""""""
""""""""""""""""""
""""""""""""""""""
""""""""""""""""""
""""""""""""""""""



    

###
##Test on 2022 "out-sample" data
###



'''Establish API connection'''
configuration = cfbd.Configuration()
configuration.api_key['Authorization'] = '+9oTfEaNp0J+Pu8n0UYskeU0uZ19wVGtGZ1oUFY23RHHnXEj2kMW6mdJTq6oEtjt'
configuration.api_key_prefix['Authorization'] = 'Bearer'
api_instance = cfbd.BettingApi(cfbd.ApiClient(configuration))

'''Pull historical betting lines for all 2023 cfb games from Bovada'''
year=2022
season_type='regular'

try:
    '''Betting lines'''
    api_response = api_instance.get_lines(year=year, season_type=season_type)
except ApiException as e:
    print("Exception when calling BettingApi->get_lines: %s\n" % e)


'''convert response from json to dictionary and convert to pandas dataframe'''

api_response_pd = pd.DataFrame.from_records([item.to_dict() for item in api_response])
api_response_pd = api_response_pd.dropna()

'''Now extract Bovada Ml from dictionary nested in list and re-insert into pandas dataframe'''

api_response_pd =api_response_pd[api_response_pd['lines'].map(len) > 0]
api_response_pd.reset_index(drop=True,inplace=True)

count = 0
betting_lines_list = list()
favorite_list = list()
for i in api_response_pd['lines']:
    temp_list = list(filter(lambda d: d['provider']== 'Bovada',api_response_pd['lines'][count]))
    fav_list = list(filter(lambda d: d['provider']== 'Bovada',api_response_pd['lines'][count]))
    if len(temp_list) > 0:
        temp_list = [-1*abs(temp_list[0]['spread'])]
        spread_formatted = fav_list[0]['formatted_spread']
        remove_digits = ''.join([i for i in spread_formatted if not i.isdigit()])
        remove_digits = remove_digits.replace(' -.','')
        remove_digits = remove_digits.replace(' -','')
        
    else:
        temp_list = temp_list
        fav_list = fav_list
        
    betting_lines_list.append(temp_list)
    favorite_list.append(remove_digits)
    count +=1



'''Create Favorite Column'''

api_response_pd['lines']= betting_lines_list
api_response_pd['favorite'] = favorite_list
api_response_pd =api_response_pd[api_response_pd['lines'].map(len) > 0]
api_response_pd= api_response_pd.dropna()
api_response_pd.reset_index(drop=True,inplace=True)


    
formatted_lines_list = list()
# formatted_favorites_list = list()

for i in api_response_pd['lines']:
    formatted_lines_list.append(i[0])
    
# for i in api_response_pd['favorite']:
#     formatted_favorites_list.append(i[0])

api_response_pd['lines'] = formatted_lines_list
# api_response_pd['favorite'] = formatted_favorites_list






# '''Establish API connection'''
# configuration = cfbd.Configuration()
# configuration.api_key['Authorization'] = '+9oTfEaNp0J+Pu8n0UYskeU0uZ19wVGtGZ1oUFY23RHHnXEj2kMW6mdJTq6oEtjt'
# configuration.api_key_prefix['Authorization'] = 'Bearer'
# api_instance = cfbd.BettingApi(cfbd.ApiClient(configuration))

# '''Pull historical betting lines for all 2023 cfb games from Bovada'''
# year=2022
# season_type='regular'

# try:
#     '''Betting lines'''
#     api_response = api_instance.get_lines(year=year, season_type=season_type)
# except ApiException as e:
#     print("Exception when calling BettingApi->get_lines: %s\n" % e)


# '''convert response from json to dictionary and convert to pandas dataframe'''

# api_response_pd = pd.DataFrame.from_records([item.to_dict() for item in api_response])
# api_response_pd = api_response_pd.dropna()

# '''Now extract BEST Ml from dictionary nested in list and re-insert into pandas dataframe'''

# api_response_pd =api_response_pd[api_response_pd['lines'].map(len) > 0]
# api_response_pd.reset_index(drop=True,inplace=True)


# '''NOTE NO LONGER LOOKING AT ONLY BOVADA, NOW SELECTING BEST POSSIBLE LINE'''

# count = 0
# betting_lines_list = list()
# favorite_list = list()
# spread_list = list()
# for i in api_response_pd['lines']:
#     temp_list = list(filter(lambda d: len(d['provider'])>0 ,api_response_pd['lines'][count]))
#     fav_list = list(filter(lambda d: len(d['provider'])>0 ,api_response_pd['lines'][count]))
#     if len(temp_list) > 0:
#         for k in range(0,len(temp_list)):
#             if temp_list[k]['spread'] is None:
#                 pass
#             if type(temp_list[k]['spread']) == float:
#                 best_fav_line = -1 * abs(temp_list[k]['spread'])
#             else:
#                 best_fav_line = -1*max((abs(temp_list[k]['spread'])))
  
            
#             spread_formatted = fav_list[0]['formatted_spread']
#             remove_digits = ''.join([i for i in spread_formatted if not i.isdigit()])
#             remove_digits = remove_digits.replace(' -.','')
#             remove_digits = remove_digits.replace(' -','')
        
#     else:
#         temp_list = temp_list
#         fav_list = fav_list
        
#     betting_lines_list.append(best_fav_line)
#     favorite_list.append(remove_digits)
#     count +=1



# '''Create Favorite Column'''

# api_response_pd['lines']= betting_lines_list
# api_response_pd['favorite'] = favorite_list
# api_response_pd= api_response_pd.dropna()
# api_response_pd.reset_index(drop=True,inplace=True)










'''Calculate Winning team'''

count=0
winner_list=list()
for i in api_response_pd['lines']:
    if api_response_pd['home_score'][count] - api_response_pd['away_score'][count] > 0:
        winner_list.append(api_response_pd['home_team'][count])
    else:
        winner_list.append(api_response_pd['away_team'][count])
    count+=1
    
api_response_pd['winning_team']=winner_list


'''Calculate Actual Spread'''

api_response_pd['actual_spread'] = api_response_pd['home_score'] - api_response_pd['away_score']
api_response_pd['actual_spread'] = api_response_pd['actual_spread'].abs()*-1

'''Boolean Indexing'''
'''Change instance where favorite does not win to positive to reflect not covering and creating a continuous series'''
'''If favorite wins its negative if favorite loses its positive,
so if Texas A&M -15 and Texas A&M wins by 100 print -100 if they lose by 1pt print +1''
''This allows for the win/lose result to be captured as well as the spread into a single integer'''

api_response_pd.loc[api_response_pd['favorite'] != api_response_pd['winning_team'],'actual_spread']*=-1



'''Sortby game-id'''
api_response_pd.sort_values(by='id',ascending=True,inplace=True)
api_response_pd.reset_index(drop=True,inplace=True)



'''Add Factors for Gradient boosted decision Model'''


'''Establish teams   API connection'''
configuration = cfbd.Configuration()
configuration.api_key['Authorization'] = '+9oTfEaNp0J+Pu8n0UYskeU0uZ19wVGtGZ1oUFY23RHHnXEj2kMW6mdJTq6oEtjt'
configuration.api_key_prefix['Authorization'] = 'Bearer'
api_instance = cfbd.TeamsApi(cfbd.ApiClient(configuration))

'''Pull Team Matchup History'''

min_year = 2016
max_year = 2021


'''Iterate through df and insert favorites win pct for head to head over past 6 years'''

#create a function and only run once when training model

def head_to_head(min_year,max_year):
        list_fav_win_pct = list()
        favorite = str()
        count=0
        for i in api_response_pd['lines']:
            favorite = api_response_pd['favorite'][count]
            home = api_response_pd['home_team'][count]
            away = api_response_pd['away_team'][count]
            #determine which team is favorite, set that to team1 as home team
            if favorite == home:
                team1 = favorite
                team2 = away
            else:
                team1 = favorite
                team2 = home
                
            try:
                # Team matchup history
                api_response = api_instance.get_team_matchup(team1, team2, min_year=min_year, max_year=max_year)
            except ApiException as e:
                print("Exception when calling TeamsApi->get_team_matchup: %s\n" % e)
           
            
            win_pct_calc = pd.DataFrame.from_records(api_response.to_dict())
            win_pct_calc = win_pct_calc.dropna()
            if len(win_pct_calc) > 0:
                win_pct_calc = win_pct_calc['team1_wins'][0]/len(win_pct_calc)
            else:
                win_pct_calc = 0.5
            list_fav_win_pct.append(win_pct_calc)
            count+=1
        return list_fav_win_pct



'''Create new column for head to head stat'''

list_fav_win_pct = head_to_head(min_year, max_year)
api_response_pd['head_to_head_favorites_win_pct'] = list_fav_win_pct


'''Research indicates if favorite has winning record vs. opponent in the past 5 yrs they only cover spread 25% indicating spread is to aggressive
however if favorite has losing record vs opponent they cover spread about 50% means spread is roughly correctly priced'''

'''So for above any values not greater than 0.5 should be set to 0 for sharper signal and reduce noise when 
training model'''

#########
#########
####
####
api_response_pd.loc[api_response_pd['head_to_head_favorites_win_pct'] <= 0.5,'head_to_head_favorites_win_pct']=0
####
####
#########
#########

''''^^^^^^Uncomment line 180,181,190 to update the dataframe with head to head matchup ~ 12 minutes^^^^^^'''




  

   
 

'''Establish Ratings  API connection'''
configuration = cfbd.Configuration()
configuration.api_key['Authorization'] = '+9oTfEaNp0J+Pu8n0UYskeU0uZ19wVGtGZ1oUFY23RHHnXEj2kMW6mdJTq6oEtjt'
configuration.api_key_prefix['Authorization'] = 'Bearer'
api_instance = cfbd.RatingsApi(cfbd.ApiClient(configuration))




'''Iterate through df and calculate elo rating for home and away team for every week'''
'''Initial api pull outside loop but nested loop handles response'''




#create a function and only run once when training model

season_type = 'both'

def elo_rating(season_type):

        list_elo_diff_underdog_fav = list()
        game_ids = list()
        
        favorite = str()
        underdog = str()
        
        
        count=0
        for f in range(1,16):
            
            if f==1:
                week = 1
                year=2021
            if f > 1:
                week = f
                year = 2022
            else:
                week = f
                year = 2022
                
            #pull api data for favorite
            try:
                # Historical Elo ratings
                api_response = api_instance.get_elo_ratings(year=year, week=week, season_type=season_type)
                
            except ApiException as e:
                print("Exception when calling RatingsApi->get_elo_ratings: %s\n" % e)
           
            
            elo_rating = pd.DataFrame.from_records([item.to_dict() for item in api_response])
            elo_rating = elo_rating.dropna()

            
            sample_df = api_response_pd.loc[api_response_pd['week']==week]
            sample_df.reset_index(drop=True,inplace=True)
            
            count=0
            for i in sample_df['id']:
               favorite = sample_df['favorite'][count]
               home = sample_df['home_team'][count]
               away = sample_df['away_team'][count]

               #determine which team is underdog
               if favorite == home:
                   underdog = away
               else:
                   underdog = home   
                   
               elo_fav = elo_rating.loc[elo_rating['team']==favorite]['elo']
               elo_underdog = elo_rating.loc[elo_rating['team']==underdog]['elo']

               if elo_fav.empty or elo_underdog.empty:
                   elo_fav = 1500
                   elo_underdog =1500
               else:
                   elo_fav = elo_fav.values[0]
                   elo_underdog = elo_underdog.values[0]
               
                
               elo_rating_diff = elo_underdog - elo_fav
               list_elo_diff_underdog_fav.append(elo_rating_diff)
               
               game_ids.append(i)
               count+=1   
               
        return list_elo_diff_underdog_fav,game_ids        
            
           
            

            
            
            
          



'''Create new column for elo ratings stat'''

list_elo_diff_underdog_fav = elo_rating(season_type)[0]
game_ids = elo_rating(season_type)[1]

'''create a dicitonary then a sorted datframe on game-ids'''

data = {'id': game_ids, 'elo_rating_underdog_fav_diff':list_elo_diff_underdog_fav}
insert_df = pd.DataFrame(data)
insert_df.sort_values(by='id',ascending=True,inplace=True)


api_response_pd['elo_rating_underdog_fav_diff'] = insert_df['elo_rating_underdog_fav_diff']

  


'''Research indicates if underdog elo rating is greater than the favorite, 
then the favorite covers the spread 55% of the time,
and the favorite spread is mispriced most likely overcompensation for explosiveness'''


'''So for above... any values not greater than or equal to 1 should be set to 0
 for sharper signal and reduce noise when 
training model'''




#########
#########
####
####
api_response_pd.loc[api_response_pd['elo_rating_underdog_fav_diff'] < 0,'elo_rating_underdog_fav_diff']=0
####
####
#########
#########


'''^^^uncomment lines 305,306,310-312,315,330 to build the elo rating column^^^'''





start_year = 2020
end_year=2021

   

'''Establish Recruiting  API connection'''
configuration = cfbd.Configuration()
configuration.api_key['Authorization'] = '+9oTfEaNp0J+Pu8n0UYskeU0uZ19wVGtGZ1oUFY23RHHnXEj2kMW6mdJTq6oEtjt'
configuration.api_key_prefix['Authorization'] = 'Bearer'
api_instance = cfbd.RecruitingApi(cfbd.ApiClient(configuration))


'''Pulling only WR/QB positon data from 42 Analytics supports most important recruits'''


try:
    # Recruit position group ratings
    api_response = api_instance.get_recruiting_groups(start_year=start_year, end_year=end_year)
except ApiException as e:
    print("Exception when calling RecruitingApi->get_recruiting_groups: %s\n" % e)

recruit_groups_team = pd.DataFrame.from_records([item.to_dict() for item in api_response])
recruit_groups_team = recruit_groups_team.dropna()

recruit_groups_team = recruit_groups_team.loc[recruit_groups_team['position_group'].isin(['Quarterback','Receiver','Defensive Back'])]

recruit_groups_team =recruit_groups_team.groupby('team',group_keys=False)[['total_rating']].apply(lambda x: x.mean())



'''Create new column for underdog_stars and fav_stars'''

favorite = str()
underdog = str()
list_underdog=list()

list_underdog_stars = list()
list_fav_stars = list()

list_fav_minus_underdog_stars = list()

count=0
for i in api_response_pd['lines']:
    favorite = api_response_pd['favorite'][count]
    home = api_response_pd['home_team'][count]
    away = api_response_pd['away_team'][count]
    #determine which team is favorite, set that to team1 as home team
    if favorite == home:
        underdog = away
    else:
        underdog=home

    if len(recruit_groups_team[recruit_groups_team.index==underdog]) > 0 and len(recruit_groups_team[recruit_groups_team.index==favorite]) > 0:
        selected_underdog = recruit_groups_team[recruit_groups_team.index==underdog].values[0][0]
        selected_fav = recruit_groups_team[recruit_groups_team.index==favorite].values[0][0]
        
        list_underdog_stars.append(selected_underdog)
        list_fav_stars.append(selected_fav)
        
    else:
        list_underdog_stars.append(0)
        list_fav_stars.append(0)
    list_underdog.append(underdog)    
    count+=1

list_fav_minus_underdog_stars = np.subtract(list_fav_stars,list_underdog_stars).tolist()

'''Add new Column for difference of favorite recruit position rank vs underdog recruit position rank'''

api_response_pd['favorite_minus_underdog_stars'] = list_fav_minus_underdog_stars


'''Create Underdog Column'''

api_response_pd['underdog'] = list_underdog
api_response_pd= api_response_pd.dropna()
api_response_pd.reset_index(drop=True,inplace=True)







'''no need to adjust.. note that any teams that have no recruits have a 0 score'''

'''Research suggest that when the underdog has better qb/lb then the favorite only covers 46% of the time'''
'''Research suggest that when the underdog has better qb/wr then the favorite only covers 43% of the time'''



###Stopped here, need to pull in stats should I use returning production, its only ppa??
###should test out both



'''Create feature for weekday of game as a variable 1 = sunday and 7= saturday'''
api_response_pd['start_day_of_week'] = pd.to_datetime(api_response_pd['start_date']).dt.weekday

'''Research shows games on Thur or Friday favorite has a 57% of covering
   games on sunday the favorite chance to cover drops to 45%'''


'''Set non Sunday,Thur,or Fri games to 0 for day of week variable'''

api_response_pd.loc[~api_response_pd['start_day_of_week'].isin([6,4,3]),'start_day_of_week']=0








'''Research shows only a few teams have actually stats sig home feilds here are the ones selected:
    Michigan,Penn State, Ohio State, Texas A&M, Florida, LSU, USC, Texas,Clemson,etc'''


'''When one of these teams is favorite and at home they have a 50% chance of winning,
when underdogs they  at home sample HAVE A 62% OF COVERING SPREAD'''


'''Manual create column for super location integer home/away data, if home =1, else =0'''

super_locations =['Texas A&M','LSU','Ohio State','Georgia', 'Penn State', 'Wisconsin',
                  'Oklahoma', 'Florida State','Florida', 'Oregon', 'Clemson', 'Tennessee',
                  'Auburn', 'South Carolina', 'Michigan', 'USC']


#create zero filled columns
api_response_pd['fav_at_home_integer'] =0
api_response_pd['underdog_at_home_integer'] = 0


#assign 1 if fav home or underdog at home
api_response_pd.loc[api_response_pd['favorite'] == api_response_pd['home_team'],'fav_at_home_integer']=1
api_response_pd.loc[api_response_pd['favorite'] != api_response_pd['home_team'],'underdog_at_home_integer']=1

#change the 1 to 0 if team is not a super_location
api_response_pd.loc[~api_response_pd['favorite'].isin(super_locations),'fav_at_home_integer']=0
api_response_pd.loc[~api_response_pd['favorite'].isin(super_locations),'underdog_at_home_integer']=0





#########
#########
####
####
api_response_pd.loc[api_response_pd['elo_rating_underdog_fav_diff'] < 0,'elo_rating_underdog_fav_diff']=0
####
####
#########
#########






'''Establish stats API connection'''
configuration = cfbd.Configuration()
configuration.api_key['Authorization'] = '+9oTfEaNp0J+Pu8n0UYskeU0uZ19wVGtGZ1oUFY23RHHnXEj2kMW6mdJTq6oEtjt'
configuration.api_key_prefix['Authorization'] = 'Bearer'
api_instance = cfbd.StatsApi(cfbd.ApiClient(configuration))


'''Pull historical all possible stats for every team for the 2022 cfb season, use for start of next season'''
year=2021
exclude_garbage_time = True

try:
    '''stats'''
    api_response = api_instance.get_advanced_team_season_stats(year=year, exclude_garbage_time=exclude_garbage_time)
except ApiException as e:
    print("Exception when calling StatsApi->get_advanced_team_season_stats: %s\n" % e)



'''convert response from json to dictionary and convert to pandas dataframe'''

api_stats = pd.DataFrame.from_records([item.to_dict() for item in api_response])
api_stats = api_stats.dropna()




'''Filter Team stats on unique Teams list'''

unique_home_team_names = api_response_pd['home_team'].unique().tolist()
unique_away_team_names = api_response_pd['away_team'].unique().tolist()
total_team_names = unique_home_team_names+unique_away_team_names

api_stats = api_stats[api_stats['team'].isin(total_team_names)]




'''uncompress dictionary that is nested in pandas dataframe
    create individual stats columns for offense and defense'''
    
    
'''layer1 and layer2 stats layer 1 are used as is, layer2 requires expanding dictionary'''
    
stats_layer1 = ['plays', 'drives', 'ppa', 'total_ppa',
'success_rate', 'explosiveness', 'power_success', 'stuff_rate', 
'line_yards', 'line_yards_total', 'second_level_yards', 
'second_level_yards_total','open_field_yards', 'open_field_yards_total', 
'total_opportunies', 'points_per_opportunity']

#    
'''new calc called chaotic effiecentcy index'''
#offense = havoc_frontseven +standarddowns success rate+(passing downs explos+passsing plays explo)*passing plays success rate
# + (rushing down explo+ppa_rushing_plays)*rushing plays success rate

#defense = havoc_frontseven +standarddowns success rate+(passing downs explos+passsing plays explo)*passing plays success rate
# + (rushing down explo+ppa_rushing_plays)*rushing plays success rate
 
    


'''Offense & defense stats all of prior season'''
'''Need to repeat with retutning players for sharper numbers'''

api_stats.reset_index(drop=True,inplace=True)

df_ex_database = []
count = 0     
for k in api_stats['team']:
    ##see if there is a way to append data to pd.df line by line to 'ex_dataset' I think I can do this using dictionary and.loc
    
    layer1_row_as_dict_type = {'team': k}
    
    #offense
    for n in stats_layer1:
      new_str = 'offense_'+n
      #print(new_str,n)
      layer1_row_as_dict_type[new_str] =api_stats['offense'][count][n]
    
    #defense
    for n in stats_layer1:
      new_str = 'defense_'+n
      layer1_row_as_dict_type[new_str] =api_stats['defense'][count][n]
    
     #offense chaotic eff index(OCEI) 
    ocei = api_stats['offense'][count]['havoc']['front_seven'] 
    ocei += api_stats['offense'][count]['standard_downs']['success_rate']
    ocei += ((api_stats['offense'][count]['passing_downs']['explosiveness']+api_stats['offense'][count]['passing_plays']['explosiveness'])*api_stats['offense'][count]['passing_plays']['success_rate'])
    ocei +=  ((api_stats['offense'][count]['standard_downs']['explosiveness']+api_stats['offense'][count]['rushing_plays']['ppa'])*api_stats['offense'][count]['rushing_plays']['success_rate'])
    
    #offense chaotic eff index(DCEI) 
    dcei = api_stats['defense'][count]['havoc']['front_seven'] 
    dcei -= api_stats['defense'][count]['standard_downs']['success_rate']
    dcei -= ((api_stats['defense'][count]['passing_downs']['explosiveness']+api_stats['defense'][count]['passing_plays']['explosiveness'])*api_stats['defense'][count]['passing_plays']['success_rate'])
    dcei -=  ((api_stats['defense'][count]['standard_downs']['explosiveness']+api_stats['defense'][count]['rushing_plays']['ppa'])*api_stats['defense'][count]['rushing_plays']['success_rate'])
   
    layer1_row_as_dict_type['OCEI'] = ocei
    layer1_row_as_dict_type['DCEI'] = dcei   
    
    df_ex_database.append(layer1_row_as_dict_type)  
    count+=1
    

    
    
df_ex_database = pd.DataFrame.from_dict(df_ex_database, orient='columns')



#merge ex_dataset and api_response_pd on home_team name and team name


##then rename all column stats to home_off_stats
##then repeat merge and rename with away_stats

###create OCEI and DCEI difference stat, leave reminder as outright vals


'''Merge ex_dataset to main dataframe on the key 'favorite' column
rename column for merge operation
rename all columns to fav_stat_name
repeat above for underdog and then create a diff column
'''
df_ex_database.rename(columns={'team':'favorite'},inplace = True)
merged_api = pd.merge(api_response_pd,df_ex_database, on='favorite')

merged_api.rename(columns={'offense_plays':'fav_offense_plays', 'offense_drives':'fav_offense_drives', 
                           'offense_ppa':'fav_offense_ppa', 'offense_total_ppa':'fav_offense_total_ppa',
                           'offense_success_rate': 'fav_offense_success_rate','offense_success_rate':'fav_offense_success_rate',
                           'offense_explosiveness': 'fav_offense_explosiveness','offense_power_success':'fav_offense_power_success',
                           'offense_stuff_rate':'fav_offense_stuff_rate', 'offense_line_yards':'fav_offense_line_yards',
                           'offense_line_yards_total':'fav_offense_line_yards_total','offense_second_level_yards':'fav_offense_second_level_yards',
                           'offense_second_level_yards_total':'fav_offense_second_level_yards_total', 
                           'offense_open_field_yards':'fav_offense_open_field_yards','offense_open_field_yards_total':'fav_offense_open_field_yards_total',
                           'offense_total_opportunies':'fav_offense_total_opportunies','offense_points_per_opportunity':'fav_offense_points_per_opportunity',
                           'defense_plays': 'fav_defense_plays','defense_drives': 'fav_defense_drives','defense_ppa': 'fav_defense_ppa',
                           'defense_total_ppa':'fav_defense_total_ppa', 'defense_success_rate':'fav_defense_success_rate',
                           'defense_explosiveness':'fav_defense_explosiveness','defense_power_success': 'fav_defense_power_success', 
                           'defense_stuff_rate':'fav_defense_stuff_rate','defense_line_yards':'fav_defense_line_yards',
                           'defense_line_yards_total':'fav_defense_line_yards_total', 'defense_second_level_yards':'fav_defense_second_level_yards',
                           'defense_second_level_yards_total': 'fav_defense_second_level_yards_total','defense_open_field_yards':'fav_defense_open_field_yards',
                           'defense_open_field_yards_total':'fav_defense_open_field_yards_total','defense_total_opportunies':'fav_defense_total_opportunies',
                           'defense_points_per_opportunity':'fav_defense_points_per_opportunity', 'OCEI':'fav_OCEI','DCEI':'fav_DCEI'},inplace = True)


merged_api= merged_api.fillna(0)
merged_api.reset_index(drop=True,inplace=True)






df_ex_database.rename(columns={'favorite':'underdog'},inplace = True)
merged_api = pd.merge(merged_api,df_ex_database, on='underdog')

merged_api.rename(columns={'offense_plays':'underdog_offense_plays', 'offense_drives':'underdog_offense_drives', 
                           'offense_ppa':'underdog_offense_ppa', 'offense_total_ppa':'underdog_offense_total_ppa',
                           'offense_success_rate': 'underdog_offense_success_rate','offense_success_rate':'underdog_offense_success_rate',
                           'offense_explosiveness': 'underdog_offense_explosiveness','offense_power_success':'underdog_offense_power_success',
                           'offense_stuff_rate':'underdog_offense_stuff_rate', 'offense_line_yards':'underdog_offense_line_yards',
                           'offense_line_yards_total':'underdog_offense_line_yards_total','offense_second_level_yards':'underdog_offense_second_level_yards',
                           'offense_second_level_yards_total':'underdog_offense_second_level_yards_total', 
                           'offense_open_field_yards':'underdog_offense_open_field_yards','offense_open_field_yards_total':'underdog_offense_open_field_yards_total',
                           'offense_total_opportunies':'underdog_offense_total_opportunies','offense_points_per_opportunity':'underdog_offense_points_per_opportunity',
                           'defense_plays': 'underdog_defense_plays','defense_drives': 'underdog_defense_drives','defense_ppa': 'underdog_defense_ppa',
                           'defense_total_ppa':'underdog_defense_total_ppa', 'defense_success_rate':'underdog_defense_success_rate',
                           'defense_explosiveness':'underdog_defense_explosiveness','defense_power_success': 'underdog_defense_power_success', 
                           'defense_stuff_rate':'underdog_defense_stuff_rate','defense_line_yards':'underdog_defense_line_yards',
                           'defense_line_yards_total':'underdog_defense_line_yards_total', 'defense_second_level_yards':'underdog_defense_second_level_yards',
                           'defense_second_level_yards_total': 'underdog_defense_second_level_yards_total','defense_open_field_yards':'underdog_defense_open_field_yards',
                           'defense_open_field_yards_total':'underdog_defense_open_field_yards_total','defense_total_opportunies':'underdog_defense_total_opportunies',
                           'defense_points_per_opportunity':'underdog_defense_points_per_opportunity', 'OCEI':'underdog_OCEI','DCEI':'underdog_DCEI'},inplace = True)


merged_api= merged_api.fillna(0)
merged_api.reset_index(drop=True,inplace=True)


"""ATTENTION: API_RESPONSE_PD IS NOW MERGED_API """



'''Transfer Portal Stats'''

year = 2022

   

'''Establish Transfer Portal  API connection'''
configuration = cfbd.Configuration()
configuration.api_key['Authorization'] = '+9oTfEaNp0J+Pu8n0UYskeU0uZ19wVGtGZ1oUFY23RHHnXEj2kMW6mdJTq6oEtjt'
configuration.api_key_prefix['Authorization'] = 'Bearer'
api_instance = cfbd.PlayersApi(cfbd.ApiClient(configuration))


'''Pulling all positions on hypothesis that transfer players are likely to start immediately'''


try:
    # Transfer portal by season
    api_response = api_instance.get_transfer_portal(year)
except ApiException as e:
    print("Exception when calling PlayersApi->get_transfer_portal: %s\n" % e)

transfer_portal_team = pd.DataFrame.from_records([item.to_dict() for item in api_response])
transfer_portal_team= transfer_portal_team.dropna()

'''Only selected transfers that have immediate eligibility'''

transfer_portal_team= transfer_portal_team.loc[transfer_portal_team['eligibility']=='Immediate']


'''Might not have a position effect like recruiting'''
#transfer_portal_team= transfer_portal_team.loc[transfer_portal_team['position'].isin(['QB','WR','CB','S'])]

transfer_portal_team_new =transfer_portal_team.groupby('destination',group_keys=False)[['stars']].apply(lambda x: x.sum())
transfer_portal_team_exit = transfer_portal_team.groupby('origin',group_keys=False)[['stars']].apply(lambda x: x.sum())

transfer_portal_team_net = transfer_portal_team_new.sub(transfer_portal_team_exit, fill_value=0).fillna(0)

'''Create new column for underdog_transfer and fav_transfer'''

favorite = str()
underdog = str()

list_underdog_transfer = list()
list_fav_transfer = list()

list_fav_minus_underdog_transfer = list()

count=0
for i in merged_api['lines']:
    favorite = merged_api['favorite'][count]
    home = merged_api['home_team'][count]
    away = merged_api['away_team'][count]
    #determine which team is favorite, set that to team1 as home team
    if favorite == home:
        underdog = away
    else:
        underdog=home

    if len(transfer_portal_team_net[transfer_portal_team_net.index==underdog]) > 0 and len(transfer_portal_team_net[transfer_portal_team_net.index==favorite]) > 0:
        selected_underdog = transfer_portal_team_net[transfer_portal_team_net.index==underdog].values[0][0]
        selected_fav = transfer_portal_team_net[transfer_portal_team_net.index==favorite].values[0][0]
        
        list_underdog_transfer.append(selected_underdog)
        list_fav_transfer.append(selected_fav)
        
    else:
        list_underdog_transfer.append(0)
        list_fav_transfer.append(0)
    list_underdog.append(underdog)    
    count+=1

list_fav_minus_underdog_transfer = np.subtract(list_fav_transfer,list_underdog_transfer).tolist()

'''Add new Column for difference of favorite net transfer portal - underdog net transfer portal'''



merged_api['net_transfer_portal_favorite_minus_underdog_stars'] = list_fav_minus_underdog_transfer

merged_api= merged_api.fillna(0)
merged_api.reset_index(drop=True,inplace=True)







'''Now Pull all Returning player stats'''


list_returning_production_diff = list()
favorite = str()
underdog = str()


#pull api data for favorite
try:
    # Team returning production metrics
    api_response = api_instance.get_returning_production(year=year)
except ApiException as e:
    print("Exception when calling PlayersApi->get_returning_production: %s\n" % e)
           

returning_stats = pd.DataFrame.from_records([item.to_dict() for item in api_response])
returning_stats = returning_stats[returning_stats['team'].isin(total_team_names)]    



'''Stats Layer1 there are no additional stats'''

stats_layer1 = ['total_ppa', 'total_passing_ppa',
       'total_receiving_ppa', 'total_rushing_ppa', 'percent_ppa',
       'percent_passing_ppa', 'percent_receiving_ppa', 'percent_rushing_ppa',
       'usage', 'passing_usage', 'receiving_usage', 'rushing_usage']


'''Offense & defense stats all of prior season'''
'''Need to repeat with retutning players for sharper numbers'''

returning_stats.reset_index(drop=True,inplace=True)



#########
#########Break, above is clean code
#########




'''Merge returning_stats to main dataframe on the key 'favorite' column
rename column for merge operation
rename all columns to fav_stat_name
repeat above for underdog
'''
returning_stats.rename(columns={'team':'favorite'},inplace = True)
merged_api = pd.merge(merged_api,returning_stats,on='favorite')

merged_api.rename(columns={'total_ppa':'fav_total_ppa', 'total_passing_ppa':'fav_total_passing_ppa',
       'total_receiving_ppa':'fav_total_receiving_ppa', 'total_rushing_ppa':'fav_total_rushing_ppa', 
       'percent_ppa':'fav_percent_ppa','percent_passing_ppa':'fav_percent_passing_ppa', 'percent_receiving_ppa':'fav_percent_receiving_ppa',
       'percent_rushing_ppa':'fav_percent_rushing_ppa','usage':'fav_usage', 'passing_usage':'fav_passing_usage',
       'receiving_usage':'fav_receiving_usage', 'rushing_usage':'fav_rushing_usage'},inplace = True)


merged_api =merged_api.fillna(0)
merged_api.reset_index(drop=True,inplace=True)




returning_stats.rename(columns={'favorite':'underdog'},inplace = True)
merged_api = pd.merge(merged_api,returning_stats, on = 'underdog')

merged_api.rename(columns={'total_ppa':'underdog_total_ppa', 'total_passing_ppa':'underdog_total_passing_ppa',
       'total_receiving_ppa':'underdog_total_receiving_ppa', 'total_rushing_ppa':'underdog_total_rushing_ppa', 
       'percent_ppa':'underdog_percent_ppa','percent_passing_ppa':'underdog_percent_passing_ppa', 'percent_receiving_ppa':'underdog_percent_receiving_ppa',
       'percent_rushing_ppa':'underdog_percent_rushing_ppa','usage':'underdog_usage', 'passing_usage':'underdog_passing_usage',
       'receiving_usage':'underdog_receiving_usage', 'rushing_usage':'underdog_rushing_usage'},inplace = True)


merged_api =merged_api.fillna(0)
merged_api.reset_index(drop=True,inplace=True)




'''RENAME MERGED_API TO FINAL_DATA'''
final_data = merged_api

'''Create target column that equals diff between betting line-actual'''
final_data['target'] = final_data['lines']-final_data['actual_spread']



'''Format Data'''
final_data = final_data.loc[final_data['lines']>=-16]
final_data = final_data.fillna(0)
final_data.reset_index(drop=True,inplace=True)








"""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""
'''USE COEFS FROM RIDGE REGRESSION MODEL'''
"""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""


#matplotlib inline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import scale 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

'''We will use the sklearn package in order to perform ridge regression and the lasso.
 The main functions in this package that we care about are Ridge(), which can be 
 used to fit ridge regression models, and Lasso() which will fit lasso models. 
 They also have cross-validated counterparts: RidgeCV() and LassoCV(). We'll use these a bit later.'''


#assign target column to y variable

y = final_data.target
#y = final_data.actual_spread


#Here are all the features we will use, any calculated variables showed noticeable significance in predicitng spread
#and should be include

#remove non numeric variables, also remove betting line as that would most likely result in overfitting
#below is the feature set we will use


X = final_data[['lines','head_to_head_favorites_win_pct','elo_rating_underdog_fav_diff','favorite_minus_underdog_stars',
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
                'underdog_rushing_usage']]


X.info()


# Create a StandardScaler object
scaler = StandardScaler()

# Fit and transform the data
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

'''SEASON 2022 INTIAL SIMPLE BACKTEST'''
#pulled features + X(which is normalized) to append prediction column to final_data
backtest_2022 = final_data

#Create expected_value column using sumproduct
backtest_2022['expected_value'] = (X[model_features]*resulting_coefs).sum(axis=1)
#backtest_2022['expected_spread'] = (X[model_features]*resulting_coefs).sum(axis=1)

#Now add expected value column to lines to get projected_spread.THEN MULTIPLY BY -1 TO GET CORRECT DIRECTION, NEED TO PAST FORMULA BELOW:
    #target = lines -actual
    #expected value = lines - actual
    #expected_value -lines = -actual
    #-expected_value +lines = actual
backtest_2022['projected_spread'] = -1*backtest_2022['expected_value'] +backtest_2022['lines']


'''EXPECTED VALUE DID NOT HAVE GOOD RESULTS OUT OF SAMPLE, I belive its do tot teh model being trained withh normalized data and then 
applying this diff to an unnormalized dseries to get the projected spread, will now use a projected spread calc, unf lower win rate but more robust'''

'''Need to get math for each bet'''




###2024 spread prediction

import cfbd
from cfbd.rest import ApiException
import pandas as pd
import time
import numpy as np


'''Establish API connection'''
configuration = cfbd.Configuration()
configuration.api_key['Authorization'] = '+9oTfEaNp0J+Pu8n0UYskeU0uZ19wVGtGZ1oUFY23RHHnXEj2kMW6mdJTq6oEtjt'
configuration.api_key_prefix['Authorization'] = 'Bearer'
api_instance = cfbd.BettingApi(cfbd.ApiClient(configuration))

'''Pull historical betting lines for all 2023 cfb games from Bovada'''
year=2024
season_type='regular'
week = int(input('INPUT THE CURRENT WEEK:::  ')) ###CHANGE TO Current week game




try:
    '''Betting lines'''
    api_response = api_instance.get_lines(year=year, season_type=season_type,week=week)
except ApiException as e:
    print("Exception when calling BettingApi->get_lines: %s\n" % e)


'''convert response from json to dictionary and convert to pandas dataframe'''

api_response_pd = pd.DataFrame.from_records([item.to_dict() for item in api_response])
#api_response_pd = api_response_pd.dropna()

'''Now extract Bovada Ml from dictionary nested in list and re-insert into pandas dataframe'''

api_response_pd =api_response_pd[api_response_pd['lines'].map(len) > 0]
api_response_pd.reset_index(drop=True,inplace=True)

count = 0
betting_lines_list = list()
favorite_list = list()
for i in api_response_pd['lines']:
    temp_list = list(filter(lambda d: d['provider']== 'Bovada',api_response_pd['lines'][count]))
    fav_list = list(filter(lambda d: d['provider']== 'Bovada',api_response_pd['lines'][count]))
    if len(temp_list) > 0:
        temp_list = [-1*abs(temp_list[0]['spread'])]
        spread_formatted = fav_list[0]['formatted_spread']
        remove_digits = ''.join([i for i in spread_formatted if not i.isdigit()])
        remove_digits = remove_digits.replace(' -.','')
        remove_digits = remove_digits.replace(' -','')
        
    else:
        temp_list = temp_list
        fav_list = fav_list
        
    betting_lines_list.append(temp_list)
    favorite_list.append(remove_digits)
    count +=1



'''Create Favorite Column'''

api_response_pd['lines']= betting_lines_list
api_response_pd['favorite'] = favorite_list
api_response_pd =api_response_pd[api_response_pd['lines'].map(len) > 0]
#api_response_pd= api_response_pd.loc[api_response_pd['home_score'].isnull() == True]
#api_response_pd= api_response_pd.loc[api_response_pd['away_score'].isnull() == True]
api_response_pd.fillna(0)
api_response_pd.reset_index(drop=True,inplace=True)


    
formatted_lines_list = list()
# formatted_favorites_list = list()

for i in api_response_pd['lines']:
    formatted_lines_list.append(i[0])
    
# for i in api_response_pd['favorite']:
#     formatted_favorites_list.append(i[0])

api_response_pd['lines'] = formatted_lines_list
# api_response_pd['favorite'] = formatted_favorites_list












'''Sortby game-id'''
api_response_pd.sort_values(by='id',ascending=True,inplace=True)
api_response_pd.reset_index(drop=True,inplace=True)



'''Add Factors for Gradient boosted decision Model'''


'''Establish teams   API connection'''
configuration = cfbd.Configuration()
configuration.api_key['Authorization'] = '+9oTfEaNp0J+Pu8n0UYskeU0uZ19wVGtGZ1oUFY23RHHnXEj2kMW6mdJTq6oEtjt'
configuration.api_key_prefix['Authorization'] = 'Bearer'
api_instance = cfbd.TeamsApi(cfbd.ApiClient(configuration))

'''Pull Team Matchup History'''

min_year = 2017
max_year = 2023


'''Iterate through df and insert favorites win pct for head to head over past 6 years'''

#create a function and only run once when training model

def head_to_head(min_year,max_year):
        list_fav_win_pct = list()
        favorite = str()
        count=0
        for i in api_response_pd['lines']:
            favorite = api_response_pd['favorite'][count]
            home = api_response_pd['home_team'][count]
            away = api_response_pd['away_team'][count]
            #determine which team is favorite, set that to team1 as home team
            if favorite == home:
                team1 = favorite
                team2 = away
            else:
                team1 = favorite
                team2 = home
                
            try:
                # Team matchup history
                api_response = api_instance.get_team_matchup(team1, team2, min_year=min_year, max_year=max_year)
            except ApiException as e:
                print("Exception when calling TeamsApi->get_team_matchup: %s\n" % e)
           
            
            win_pct_calc = pd.DataFrame.from_records(api_response.to_dict())
            win_pct_calc = win_pct_calc.dropna()
            if len(win_pct_calc) > 0:
                win_pct_calc = win_pct_calc['team1_wins'][0]/len(win_pct_calc)
            else:
                win_pct_calc = 0.5
            list_fav_win_pct.append(win_pct_calc)
            count+=1
        return list_fav_win_pct



'''Create new column for head to head stat'''

list_fav_win_pct = head_to_head(min_year, max_year)
api_response_pd['head_to_head_favorites_win_pct'] = list_fav_win_pct


'''Research indicates if favorite has winning record vs. opponent in the past 5 yrs they only cover spread 25% indicating spread is to aggressive
however if favorite has losing record vs opponent they cover spread about 50% means spread is roughly correctly priced'''

'''So for above any values not greater than 0.5 should be set to 0 for sharper signal and reduce noise when 
training model'''

#########
#########
####
####
api_response_pd.loc[api_response_pd['head_to_head_favorites_win_pct'] <= 0.5,'head_to_head_favorites_win_pct']=0
#### UNCOMMENTED
####
#########
#########

''''^^^^^^Uncomment line 180,181,190 to update the dataframe with head to head matchup ~ 12 minutes^^^^^^'''




  

   
 

'''Establish Ratings  API connection'''
configuration = cfbd.Configuration()
configuration.api_key['Authorization'] = '+9oTfEaNp0J+Pu8n0UYskeU0uZ19wVGtGZ1oUFY23RHHnXEj2kMW6mdJTq6oEtjt'
configuration.api_key_prefix['Authorization'] = 'Bearer'
api_instance = cfbd.RatingsApi(cfbd.ApiClient(configuration))




'''Iterate through df and calculate elo rating for home and away team for every week'''
'''Initial api pull outside loop but nested loop handles response'''




#create a function and only run once when training model

season_type = 'both'

def elo_rating(season_type,week):

        list_elo_diff_underdog_fav = list()
        game_ids = list()
        
        favorite = str()
        underdog = str()
        
        
            
        week = week
        year = 2024
            
        #pull api data for favorite
        try:
            # Historical Elo ratings
            api_response = api_instance.get_elo_ratings(year=year, week=week, season_type=season_type)
            
        except ApiException as e:
            print("Exception when calling RatingsApi->get_elo_ratings: %s\n" % e)
       
        
        elo_rating = pd.DataFrame.from_records([item.to_dict() for item in api_response])
        elo_rating = elo_rating.dropna()

        
        sample_df = api_response_pd.loc[api_response_pd['week']==week]
        sample_df.reset_index(drop=True,inplace=True)
        
        count=0
        for i in sample_df['id']:
           favorite = sample_df['favorite'][count]
           home = sample_df['home_team'][count]
           away = sample_df['away_team'][count]

           #determine which team is underdog
           if favorite == home:
               underdog = away
           else:
               underdog = home   
               
           elo_fav = elo_rating.loc[elo_rating['team']==favorite]['elo']
           elo_underdog = elo_rating.loc[elo_rating['team']==underdog]['elo']

           if elo_fav.empty or elo_underdog.empty:
               elo_fav = 1500
               elo_underdog =1500
           else:
               elo_fav = elo_fav.values[0]
               elo_underdog = elo_underdog.values[0]
           
            
           elo_rating_diff = elo_underdog - elo_fav
           list_elo_diff_underdog_fav.append(elo_rating_diff)
           
           game_ids.append(i)
           count+=1   
               
        return list_elo_diff_underdog_fav,game_ids        
            
           
            

            
            
            
          



'''Create new column for elo ratings stat'''

list_elo_diff_underdog_fav = elo_rating(season_type,week)[0]
game_ids = elo_rating(season_type,week)[1]

'''create a dicitonary then a sorted datframe on game-ids'''

data = {'id': game_ids, 'elo_rating_underdog_fav_diff':list_elo_diff_underdog_fav}
insert_df = pd.DataFrame(data)
insert_df.sort_values(by='id',ascending=True,inplace=True)


api_response_pd['elo_rating_underdog_fav_diff'] = insert_df['elo_rating_underdog_fav_diff']

  


'''Research indicates if underdog elo rating is greater than the favorite, 
then the favorite covers the spread 55% of the time,
and the favorite spread is mispriced most likely overcompensation for explosiveness'''


'''So for above... any values not greater than or equal to 1 should be set to 0
 for sharper signal and reduce noise when 
training model'''




#########
#########
####
####
api_response_pd.loc[api_response_pd['elo_rating_underdog_fav_diff'] < 0,'elo_rating_underdog_fav_diff']=0#uncommented for backtesting
####
####
#########
#########


'''^^^uncomment lines 305,306,310-312,315,330 to build the elo rating column^^^'''





start_year = 2023
end_year=2024

   

'''Establish Recruiting  API connection'''
configuration = cfbd.Configuration()
configuration.api_key['Authorization'] = '+9oTfEaNp0J+Pu8n0UYskeU0uZ19wVGtGZ1oUFY23RHHnXEj2kMW6mdJTq6oEtjt'
configuration.api_key_prefix['Authorization'] = 'Bearer'
api_instance = cfbd.RecruitingApi(cfbd.ApiClient(configuration))


'''Pulling only WR/QB positon data from 42 Analytics supports most important recruits'''


try:
    # Recruit position group ratings
    api_response = api_instance.get_recruiting_groups(start_year=start_year, end_year=end_year)
except ApiException as e:
    print("Exception when calling RecruitingApi->get_recruiting_groups: %s\n" % e)

recruit_groups_team = pd.DataFrame.from_records([item.to_dict() for item in api_response])
recruit_groups_team = recruit_groups_team.dropna()

recruit_groups_team = recruit_groups_team.loc[recruit_groups_team['position_group'].isin(['Quarterback','Receiver','Defensive Back'])]

recruit_groups_team =recruit_groups_team.groupby('team',group_keys=False)[['total_rating']].apply(lambda x: x.mean())



'''Create new column for underdog_stars and fav_stars'''

favorite = str()
underdog = str()
list_underdog=list()

list_underdog_stars = list()
list_fav_stars = list()

list_fav_minus_underdog_stars = list()

count=0
for i in api_response_pd['lines']:
    favorite = api_response_pd['favorite'][count]
    home = api_response_pd['home_team'][count]
    away = api_response_pd['away_team'][count]
    #determine which team is favorite, set that to team1 as home team
    if favorite == home:
        underdog = away
    else:
        underdog=home

    if len(recruit_groups_team[recruit_groups_team.index==underdog]) > 0 and len(recruit_groups_team[recruit_groups_team.index==favorite]) > 0:
        selected_underdog = recruit_groups_team[recruit_groups_team.index==underdog].values[0][0]
        selected_fav = recruit_groups_team[recruit_groups_team.index==favorite].values[0][0]
        
        list_underdog_stars.append(selected_underdog)
        list_fav_stars.append(selected_fav)
        
    else:
        list_underdog_stars.append(0)
        list_fav_stars.append(0)
    list_underdog.append(underdog)    
    count+=1

list_fav_minus_underdog_stars = np.subtract(list_fav_stars,list_underdog_stars).tolist()

'''Add new Column for difference of favorite recruit position rank vs underdog recruit position rank'''

api_response_pd['favorite_minus_underdog_stars'] = list_fav_minus_underdog_stars


'''Create Underdog Column'''

api_response_pd['underdog'] = list_underdog
api_response_pd= api_response_pd.drop(columns=['home_score','away_score'])
api_response_pd.reset_index(drop=True,inplace=True)





#####BREAK






'''no need to adjust.. note that any teams that have no recruits have a 0 score'''

'''Research suggest that when the underdog has better qb/lb then the favorite only covers 46% of the time'''
'''Research suggest that when the underdog has better qb/wr then the favorite only covers 43% of the time'''



###Stopped here, need to pull in stats should I use returning production, its only ppa??
###should test out both



'''Create feature for weekday of game as a variable 1 = sunday and 7= saturday'''
api_response_pd['start_day_of_week'] = pd.to_datetime(api_response_pd['start_date']).dt.weekday

'''Research shows games on Thur or Friday favorite has a 57% of covering
   games on sunday the favorite chance to cover drops to 45%'''


'''Set non Sunday,Thur,or Fri games to 0 for day of week variable'''

#api_response_pd.loc[~api_response_pd['start_day_of_week'].isin([6,4,3]),'start_day_of_week']=0
'''UNCOMMENTED'''







'''Research shows only a few teams have actually stats sig home feilds here are the ones selected:
    Michigan,Penn State, Ohio State, Texas A&M, Florida, LSU, USC, Texas,Clemson,etc'''


'''When one of these teams is favorite and at home they have a 50% chance of winning,
when underdogs they  at home sample HAVE A 62% OF COVERING SPREAD'''


'''Manual create column for super location integer home/away data, if home =1, else =0'''

super_locations =['Texas A&M','LSU','Ohio State','Georgia', 'Penn State', 'Wisconsin',
                  'Oklahoma', 'Florida State','Florida', 'Oregon', 'Clemson', 'Tennessee',
                  'Auburn', 'South Carolina', 'Michigan', 'USC']


#create zero filled columns
api_response_pd['fav_at_home_integer'] =0
api_response_pd['underdog_at_home_integer'] = 0


#assign 1 if fav home or underdog at home
api_response_pd.loc[api_response_pd['favorite'] == api_response_pd['home_team'],'fav_at_home_integer']=1
api_response_pd.loc[api_response_pd['favorite'] != api_response_pd['home_team'],'underdog_at_home_integer']=1

#change the 1 to 0 if team is not a super_location
api_response_pd.loc[~api_response_pd['favorite'].isin(super_locations),'fav_at_home_integer']=0
api_response_pd.loc[~api_response_pd['favorite'].isin(super_locations),'underdog_at_home_integer']=0





#########
#########
####
####
api_response_pd.loc[api_response_pd['elo_rating_underdog_fav_diff'] < 0,'elo_rating_underdog_fav_diff']=0
####
####UNCOMMENTED
#########
#########






'''Establish stats API connection'''
configuration = cfbd.Configuration()
configuration.api_key['Authorization'] = '+9oTfEaNp0J+Pu8n0UYskeU0uZ19wVGtGZ1oUFY23RHHnXEj2kMW6mdJTq6oEtjt'
configuration.api_key_prefix['Authorization'] = 'Bearer'
api_instance = cfbd.StatsApi(cfbd.ApiClient(configuration))


'''Pull historical all possible stats for every team for the 2023 cfb season, use for start of next season'''
year=2024
exclude_garbage_time = True

try:
    '''stats'''
    api_response = api_instance.get_advanced_team_season_stats(year=year, exclude_garbage_time=exclude_garbage_time)
except ApiException as e:
    print("Exception when calling StatsApi->get_advanced_team_season_stats: %s\n" % e)



'''convert response from json to dictionary and convert to pandas dataframe'''

api_stats = pd.DataFrame.from_records([item.to_dict() for item in api_response])
api_stats = api_stats.dropna()




'''Filter Team stats on unique Teams list'''

unique_home_team_names = api_response_pd['home_team'].unique().tolist()
unique_away_team_names = api_response_pd['away_team'].unique().tolist()
total_team_names = unique_home_team_names+unique_away_team_names

api_stats = api_stats[api_stats['team'].isin(total_team_names)]




'''uncompress dictionary that is nested in pandas dataframe
    create individual stats columns for offense and defense'''
    
    
'''layer1 and layer2 stats layer 1 are used as is, layer2 requires expanding dictionary'''
    
stats_layer1 = ['plays', 'drives', 'ppa', 'total_ppa',
'success_rate', 'explosiveness', 'power_success', 'stuff_rate', 
'line_yards', 'line_yards_total', 'second_level_yards', 
'second_level_yards_total','open_field_yards', 'open_field_yards_total', 
'total_opportunies', 'points_per_opportunity']

#    
'''new calc called chaotic effiecentcy index'''
#offense = havoc_frontseven +standarddowns success rate+(passing downs explos+passsing plays explo)*passing plays success rate
# + (rushing down explo+ppa_rushing_plays)*rushing plays success rate

#defense = havoc_frontseven +standarddowns success rate+(passing downs explos+passsing plays explo)*passing plays success rate
# + (rushing down explo+ppa_rushing_plays)*rushing plays success rate
 
    


'''Offense & defense stats all of prior season'''
'''Need to repeat with retutning players for sharper numbers'''

api_stats.reset_index(drop=True,inplace=True)

df_ex_database = []
count = 0     
for k in api_stats['team']:
    ##see if there is a way to append data to pd.df line by line to 'ex_dataset' I think I can do this using dictionary and.loc
    
    layer1_row_as_dict_type = {'team': k}
    
    #offense
    for n in stats_layer1:
      new_str = 'offense_'+n
      #print(new_str,n)
      layer1_row_as_dict_type[new_str] =api_stats['offense'][count][n]
    
    #defense
    for n in stats_layer1:
      new_str = 'defense_'+n
      layer1_row_as_dict_type[new_str] =api_stats['defense'][count][n]
    
     #offense chaotic eff index(OCEI) 
    ocei = api_stats['offense'][count]['havoc']['front_seven'] 
    ocei += api_stats['offense'][count]['standard_downs']['success_rate']
    ocei += ((api_stats['offense'][count]['passing_downs']['explosiveness']+api_stats['offense'][count]['passing_plays']['explosiveness'])*api_stats['offense'][count]['passing_plays']['success_rate'])
    ocei +=  ((api_stats['offense'][count]['standard_downs']['explosiveness']+api_stats['offense'][count]['rushing_plays']['ppa'])*api_stats['offense'][count]['rushing_plays']['success_rate'])
    
    #offense chaotic eff index(DCEI) 
    dcei = api_stats['defense'][count]['havoc']['front_seven'] 
    dcei -= api_stats['defense'][count]['standard_downs']['success_rate']
    dcei -= ((api_stats['defense'][count]['passing_downs']['explosiveness']+api_stats['defense'][count]['passing_plays']['explosiveness'])*api_stats['defense'][count]['passing_plays']['success_rate'])
    dcei -=  ((api_stats['defense'][count]['standard_downs']['explosiveness']+api_stats['defense'][count]['rushing_plays']['ppa'])*api_stats['defense'][count]['rushing_plays']['success_rate'])
   
    layer1_row_as_dict_type['OCEI'] = ocei
    layer1_row_as_dict_type['DCEI'] = dcei   
    
    df_ex_database.append(layer1_row_as_dict_type)  
    count+=1
    

    
    
df_ex_database = pd.DataFrame.from_dict(df_ex_database, orient='columns')



#merge ex_dataset and api_response_pd on home_team name and team name


##then rename all column stats to home_off_stats
##then repeat merge and rename with away_stats

###create OCEI and DCEI difference stat, leave reminder as outright vals


'''Merge ex_dataset to main dataframe on the key 'favorite' column
rename column for merge operation
rename all columns to fav_stat_name
repeat above for underdog and then create a diff column
'''
df_ex_database.rename(columns={'team':'favorite'},inplace = True)
merged_api = pd.merge(api_response_pd,df_ex_database, on='favorite')

merged_api.rename(columns={'offense_plays':'fav_offense_plays', 'offense_drives':'fav_offense_drives', 
                           'offense_ppa':'fav_offense_ppa', 'offense_total_ppa':'fav_offense_total_ppa',
                           'offense_success_rate': 'fav_offense_success_rate','offense_success_rate':'fav_offense_success_rate',
                           'offense_explosiveness': 'fav_offense_explosiveness','offense_power_success':'fav_offense_power_success',
                           'offense_stuff_rate':'fav_offense_stuff_rate', 'offense_line_yards':'fav_offense_line_yards',
                           'offense_line_yards_total':'fav_offense_line_yards_total','offense_second_level_yards':'fav_offense_second_level_yards',
                           'offense_second_level_yards_total':'fav_offense_second_level_yards_total', 
                           'offense_open_field_yards':'fav_offense_open_field_yards','offense_open_field_yards_total':'fav_offense_open_field_yards_total',
                           'offense_total_opportunies':'fav_offense_total_opportunies','offense_points_per_opportunity':'fav_offense_points_per_opportunity',
                           'defense_plays': 'fav_defense_plays','defense_drives': 'fav_defense_drives','defense_ppa': 'fav_defense_ppa',
                           'defense_total_ppa':'fav_defense_total_ppa', 'defense_success_rate':'fav_defense_success_rate',
                           'defense_explosiveness':'fav_defense_explosiveness','defense_power_success': 'fav_defense_power_success', 
                           'defense_stuff_rate':'fav_defense_stuff_rate','defense_line_yards':'fav_defense_line_yards',
                           'defense_line_yards_total':'fav_defense_line_yards_total', 'defense_second_level_yards':'fav_defense_second_level_yards',
                           'defense_second_level_yards_total': 'fav_defense_second_level_yards_total','defense_open_field_yards':'fav_defense_open_field_yards',
                           'defense_open_field_yards_total':'fav_defense_open_field_yards_total','defense_total_opportunies':'fav_defense_total_opportunies',
                           'defense_points_per_opportunity':'fav_defense_points_per_opportunity', 'OCEI':'fav_OCEI','DCEI':'fav_DCEI'},inplace = True)


merged_api= merged_api.fillna(0)
merged_api.reset_index(drop=True,inplace=True)






df_ex_database.rename(columns={'favorite':'underdog'},inplace = True)
merged_api = pd.merge(merged_api,df_ex_database, on='underdog')

merged_api.rename(columns={'offense_plays':'underdog_offense_plays', 'offense_drives':'underdog_offense_drives', 
                           'offense_ppa':'underdog_offense_ppa', 'offense_total_ppa':'underdog_offense_total_ppa',
                           'offense_success_rate': 'underdog_offense_success_rate','offense_success_rate':'underdog_offense_success_rate',
                           'offense_explosiveness': 'underdog_offense_explosiveness','offense_power_success':'underdog_offense_power_success',
                           'offense_stuff_rate':'underdog_offense_stuff_rate', 'offense_line_yards':'underdog_offense_line_yards',
                           'offense_line_yards_total':'underdog_offense_line_yards_total','offense_second_level_yards':'underdog_offense_second_level_yards',
                           'offense_second_level_yards_total':'underdog_offense_second_level_yards_total', 
                           'offense_open_field_yards':'underdog_offense_open_field_yards','offense_open_field_yards_total':'underdog_offense_open_field_yards_total',
                           'offense_total_opportunies':'underdog_offense_total_opportunies','offense_points_per_opportunity':'underdog_offense_points_per_opportunity',
                           'defense_plays': 'underdog_defense_plays','defense_drives': 'underdog_defense_drives','defense_ppa': 'underdog_defense_ppa',
                           'defense_total_ppa':'underdog_defense_total_ppa', 'defense_success_rate':'underdog_defense_success_rate',
                           'defense_explosiveness':'underdog_defense_explosiveness','defense_power_success': 'underdog_defense_power_success', 
                           'defense_stuff_rate':'underdog_defense_stuff_rate','defense_line_yards':'underdog_defense_line_yards',
                           'defense_line_yards_total':'underdog_defense_line_yards_total', 'defense_second_level_yards':'underdog_defense_second_level_yards',
                           'defense_second_level_yards_total': 'underdog_defense_second_level_yards_total','defense_open_field_yards':'underdog_defense_open_field_yards',
                           'defense_open_field_yards_total':'underdog_defense_open_field_yards_total','defense_total_opportunies':'underdog_defense_total_opportunies',
                           'defense_points_per_opportunity':'underdog_defense_points_per_opportunity', 'OCEI':'underdog_OCEI','DCEI':'underdog_DCEI'},inplace = True)


merged_api= merged_api.fillna(0)
merged_api.reset_index(drop=True,inplace=True)


"""ATTENTION: API_RESPONSE_PD IS NOW MERGED_API """



'''Transfer Portal Stats'''

year = 2024

   

'''Establish Transfer Portal  API connection'''
configuration = cfbd.Configuration()
configuration.api_key['Authorization'] = '+9oTfEaNp0J+Pu8n0UYskeU0uZ19wVGtGZ1oUFY23RHHnXEj2kMW6mdJTq6oEtjt'
configuration.api_key_prefix['Authorization'] = 'Bearer'
api_instance = cfbd.PlayersApi(cfbd.ApiClient(configuration))


'''Pulling all positions on hypothesis that transfer players are likely to start immediately'''


try:
    # Transfer portal by season
    api_response = api_instance.get_transfer_portal(year)
except ApiException as e:
    print("Exception when calling PlayersApi->get_transfer_portal: %s\n" % e)

transfer_portal_team = pd.DataFrame.from_records([item.to_dict() for item in api_response])
transfer_portal_team= transfer_portal_team.dropna()

'''Only selected transfers that have immediate eligibility'''

transfer_portal_team= transfer_portal_team.loc[transfer_portal_team['eligibility']=='Immediate']


'''Might not have a position effect like recruiting'''
#transfer_portal_team= transfer_portal_team.loc[transfer_portal_team['position'].isin(['QB','WR','CB','S'])]

transfer_portal_team_new =transfer_portal_team.groupby('destination',group_keys=False)[['stars']].apply(lambda x: x.sum())
transfer_portal_team_exit = transfer_portal_team.groupby('origin',group_keys=False)[['stars']].apply(lambda x: x.sum())

transfer_portal_team_net = transfer_portal_team_new.sub(transfer_portal_team_exit, fill_value=0).fillna(0)

'''Create new column for underdog_transfer and fav_transfer'''

favorite = str()
underdog = str()

list_underdog_transfer = list()
list_fav_transfer = list()

list_fav_minus_underdog_transfer = list()

count=0
for i in merged_api['lines']:
    favorite = merged_api['favorite'][count]
    home = merged_api['home_team'][count]
    away = merged_api['away_team'][count]
    #determine which team is favorite, set that to team1 as home team
    if favorite == home:
        underdog = away
    else:
        underdog=home

    if len(transfer_portal_team_net[transfer_portal_team_net.index==underdog]) > 0 and len(transfer_portal_team_net[transfer_portal_team_net.index==favorite]) > 0:
        selected_underdog = transfer_portal_team_net[transfer_portal_team_net.index==underdog].values[0][0]
        selected_fav = transfer_portal_team_net[transfer_portal_team_net.index==favorite].values[0][0]
        
        list_underdog_transfer.append(selected_underdog)
        list_fav_transfer.append(selected_fav)
        
    else:
        list_underdog_transfer.append(0)
        list_fav_transfer.append(0)
    list_underdog.append(underdog)    
    count+=1

list_fav_minus_underdog_transfer = np.subtract(list_fav_transfer,list_underdog_transfer).tolist()

'''Add new Column for difference of favorite net transfer portal - underdog net transfer portal'''



merged_api['net_transfer_portal_favorite_minus_underdog_stars'] = list_fav_minus_underdog_transfer

merged_api= merged_api.fillna(0)
merged_api.reset_index(drop=True,inplace=True)







'''Now Pull all Returning player stats'''


list_returning_production_diff = list()
favorite = str()
underdog = str()


#pull api data for favorite
try:
    # Team returning production metrics
    api_response = api_instance.get_returning_production(year=year)
except ApiException as e:
    print("Exception when calling PlayersApi->get_returning_production: %s\n" % e)
           

returning_stats = pd.DataFrame.from_records([item.to_dict() for item in api_response])
returning_stats = returning_stats[returning_stats['team'].isin(total_team_names)]    



'''Stats Layer1 there are no additional stats'''

stats_layer1 = ['total_ppa', 'total_passing_ppa',
       'total_receiving_ppa', 'total_rushing_ppa', 'percent_ppa',
       'percent_passing_ppa', 'percent_receiving_ppa', 'percent_rushing_ppa',
       'usage', 'passing_usage', 'receiving_usage', 'rushing_usage']


'''Offense & defense stats all of prior season'''
'''Need to repeat with retutning players for sharper numbers'''

returning_stats.reset_index(drop=True,inplace=True)



#########
#########Break, above is clean code
#########




'''Merge returning_stats to main dataframe on the key 'favorite' column
rename column for merge operation
rename all columns to fav_stat_name
repeat above for underdog
'''
returning_stats.rename(columns={'team':'favorite'},inplace = True)
merged_api = pd.merge(merged_api,returning_stats,on='favorite')

merged_api.rename(columns={'total_ppa':'fav_total_ppa', 'total_passing_ppa':'fav_total_passing_ppa',
       'total_receiving_ppa':'fav_total_receiving_ppa', 'total_rushing_ppa':'fav_total_rushing_ppa', 
       'percent_ppa':'fav_percent_ppa','percent_passing_ppa':'fav_percent_passing_ppa', 'percent_receiving_ppa':'fav_percent_receiving_ppa',
       'percent_rushing_ppa':'fav_percent_rushing_ppa','usage':'fav_usage', 'passing_usage':'fav_passing_usage',
       'receiving_usage':'fav_receiving_usage', 'rushing_usage':'fav_rushing_usage'},inplace = True)


merged_api =merged_api.fillna(0)
merged_api.reset_index(drop=True,inplace=True)




returning_stats.rename(columns={'favorite':'underdog'},inplace = True)
merged_api = pd.merge(merged_api,returning_stats, on = 'underdog')

merged_api.rename(columns={'total_ppa':'underdog_total_ppa', 'total_passing_ppa':'underdog_total_passing_ppa',
       'total_receiving_ppa':'underdog_total_receiving_ppa', 'total_rushing_ppa':'underdog_total_rushing_ppa', 
       'percent_ppa':'underdog_percent_ppa','percent_passing_ppa':'underdog_percent_passing_ppa', 'percent_receiving_ppa':'underdog_percent_receiving_ppa',
       'percent_rushing_ppa':'underdog_percent_rushing_ppa','usage':'underdog_usage', 'passing_usage':'underdog_passing_usage',
       'receiving_usage':'underdog_receiving_usage', 'rushing_usage':'underdog_rushing_usage'},inplace = True)


merged_api =merged_api.fillna(0)
merged_api.reset_index(drop=True,inplace=True)




'''RENAME MERGED_API TO FINAL_DATA'''
final_data = merged_api

'''Create target column that equals diff between betting line-actual'''
#final_data['target'] = final_data['lines']-final_data['actual_spread']



'''Format Data'''
final_data = final_data.loc[final_data['lines']>=-16]
final_data = final_data.fillna(0)
final_data.reset_index(drop=True,inplace=True)








"""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""
'''USE COEFS FROM RIDGE REGRESSION MODEL'''
"""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""


#matplotlib inline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import scale 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

'''We will use the sklearn package in order to perform ridge regression and the lasso.
 The main functions in this package that we care about are Ridge(), which can be 
 used to fit ridge regression models, and Lasso() which will fit lasso models. 
 They also have cross-validated counterparts: RidgeCV() and LassoCV(). We'll use these a bit later.'''




#Here are all the features we will use, any calculated variables showed noticeable significance in predicitng spread
#and should be include

#remove non numeric variables, also remove betting line as that would most likely result in overfitting
#below is the feature set we will use


X = final_data[['lines','head_to_head_favorites_win_pct','elo_rating_underdog_fav_diff','favorite_minus_underdog_stars',
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
                'underdog_rushing_usage']]


X.info()


# Create a StandardScaler object
scaler = StandardScaler()

# Fit and transform the data
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

'''SEASON 2022 INTIAL SIMPLE BACKTEST'''
#pulled features + X(which is normalized) to append prediction column to final_data
predict_2024= final_data

#Create expected_value column using sumproduct
predict_2024['expected_value'] = (X[model_features]*resulting_coefs).sum(axis=1)
#predict_2024['expected_spread'] = (X[model_features]*resulting_coefs).sum(axis=1)

predict_2024['projected_spread'] = -1*predict_2024['expected_value'] +predict_2024['lines']

predict_2024['confidence']=predict_2024['expected_value'].abs()

predict_2024.sort_values(by ='confidence', ascending = False ,inplace = True)


#drop all predicted spreads with favorite winning more than by 3, project upsets only
predict_2024 = predict_2024.head(5)
predict_2024 = predict_2024.loc[predict_2024['projected_spread']>-3]
predict_2024.reset_index(drop=True,inplace=True)

#display top 5 picks of the week ranked by confidence



count = 0
print()
print('Top 5 picks of the week ')
for i in predict_2024['projected_spread']:
    spread_ev = abs(i)
    line = predict_2024['lines'][count]
    favorite = predict_2024['favorite'][count]
    underdog = predict_2024['underdog'][count]
    if i < predict_2024['lines'][count]:
        cover = favorite
        non_cover =underdog
    else:
        cover = underdog
        non_cover= favorite
    if i < 0:
        winner = favorite
        loser = underdog
    else:
        winner = underdog
        loser = favorite
    print(count+1, '. ', 'Take ', cover, ' to cover the spread against', non_cover,'\n Spread is ',favorite,'by ', line,'\n Expected spread is ',
          winner, ' -', spread_ev)
    count+=1







###NEED TO DISPLAY BACKTEST STATS FOR 2022 AND 2023















#Now Test on all 2022 season need to repull data, then have to normalize and can then run




# count = 0
# expected_value = int()
# expected_value_list = list()


# for i in backtest['target']:
    


###Now add expected_value_list to new column and then add expected value to betting lines and create column called projected spread
###if projected spread is less than betting line assign 1 and if greater than assign 0, calculate sum of column and divide by len to get win rate


##create a hybrid calc for layer2 that is all in 1, normalize(cut off tails) by indexing from 10 to -10,
###sell a       b  

    
    

#Need to add expanded stats with chose the one with highest rate
# calculting a run and pass eff then ppa*explosiveness  dont include success rate'''



##add a







































print("--- %s seconds ---" % (time.time() - start_time))











