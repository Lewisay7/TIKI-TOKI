import mod_model 
import ad_model
import pandas as pd

mod =  mod_model.mod_final
ads = ad_model.ad_final

ads.index = range(1,len(ads)+1)
mod.index = range(1,len(mod)+1)

merged_data = pd.DataFrame(columns=["ad_id", "ad_score", "mod_id", "mod_score", "score_diff"])
row_count = 1
#matching advertisement to moderators
for i in range(1,len(ads)+1):
    country = ads["queue_market"][i][0]
    for j in range(1,len(mod)+1):
        if country in mod["market"][j]:
            new_data = [ads["ad_id"][i], ads["scores"][i], mod["mod_id"][j], mod["scores"][j], ads["scores"][i]-mod["scores"][j]]
            merged_data.loc[row_count] = new_data
            row_count += 1
            mod = mod.drop(axis=0, index=j)
            mod.index = range(1,len(mod)+1)
            break
        else:
            continue
        
#merge data to output difference in the scores
merged_data