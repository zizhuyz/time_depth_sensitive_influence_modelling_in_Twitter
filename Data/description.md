Below are the descriptions of datasets used in paper *"Learning Influence Probabilities and Modelling Influence Diffusion in Twitter"*.

##user_ids.csv
**uid**: a user's id given by Twitter
**nuid**: a user's id assign by myself

##status.csv
**t_id**: id given by Twitter
**timestamp_ms**: post time of *"t_id"*
**uid**: a user's id given by Twitter
**nuid**: a user's id assign by myself
**is_reply**: whether this message is a reply to another, yes:1, no:0
**is_retweet**: whether this message is a retweet to another, yes:1, no:0
**is_quote**: whether this message is a quote to another, yes:1, no:0
**replied_uid**,**replied_t_id**: user's id and message's id of the replied message
**retweeted_uid**,**retweeted_t_id**: user's id and message's id of the retweeted message
**quoted_uid**,**quoted_t_id**: user's id and message's id of the quoted message

##likes.csv
**t_id_u**: message posted by user *'u'* and liked by user *'v'*
**created_at**: post time of *"t_id_u"*
**author_uid_u**: the influencer *u*'s id given by Twitter
**liked_by_uid_v**: the influencee *v*'s id given by Twitter
**interaction_type**: 5, which is a "like" interaction

##interaction_network.csv
**v**: nuid of the influencee, the user who is influenced
**u**: nuid of the influencer, the user who influences *'v'*
**t_id_v**: message id posted by *'v'* given by Twitter
**t_id_u**: message id posted by *'u'* given by Twitter
**interaction_type**: explicit-retweet:1, reply:2, quote:3, like:5; implicit- similar content:6
**t_time_v**: post time of *'t_id_v'*
**t_time_u**: post time of *'t_id_u'*

##following_network.csv
**nuid_followee_u**: nuid of the user *'u'* who is followed by *'v'*
**nuid_follower_v**: nuid of the user *'v'* who follows *'u'*


##Paper details
Zizhu Zhang, Weiliang Zhao and Jian Yang and Cecile Paris and Surya Nepal.2019. Learning Influence Probabilities and Modelling Influence Diffusion inTwitter. InCompanion Proceedings of the 2019 World Wide Web Conference(WWW’19 Companion), May 13–17, 2019, San Francisco, CA, USA.ACM, NewYork, NY, USA, 8 pages. [https://doi.org/10.1145/3308560.3316701](https://doi.org/10.1145/3308560.3316701)

Credit to: Zizhu Zhang, [zizhu07.zhang@gmail.com](zizhu07.zhang@gmail.com)