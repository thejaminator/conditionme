## Introduction
`conditionme` is library for easily retraining existing language models to work in a conditional /decision tranformer / upside down rl fashion.

We eventually hope it can be something similar to [trl](https://github.com/lvwerra/trl), just that instead of PPO we'll train in a decision transformer fashion.
This still a very early stage library, so expect bugs and missing features.

## Why does this library exist?
I haven't found a library that allows you to easily retrain existing language models (e.g. gpt2, gpt-j) to work in a  conditional / decision tranformer / upside down rl fashion.
This library helps your easily specify a scalar target reward when training your model in this fashion. 

Most libraries for decision transformers focus on training in a game / gym environment.

There could be some aspects for training in a decision transformer fashion that could be useful for AI safety. See [Safety considerations for online generative modelling](https://www.lesswrong.com/posts/BMfNu82iunjqKyQA9/safety-considerations-for-online-generative-modeling#Safety_advantages_of_generative_modeling), [Soft optimization makes the value target bigger](https://www.lesswrong.com/posts/9fL22eBJMtyCLvL7j/soft-optimization-makes-the-value-target-bigger#Fine_tuned_generative_models), [RLHF bad, conditioning good](https://www.lesswrong.com/posts/AXpXG9oTiucidnqPK/take-13-rlhf-bad-conditioning-good)


This library helps you investigate decision transformers empirically by:
1. Providing a compatible tokenizer - what we'll call a DecisionTokenizer. Among other things, it reduces `model_max_length` by 1 so that we can reserve the first token for the reward token.  
```python
from transformers import AutoTokenizer
from conditionme import create_decision_tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
decision_tokenizer = create_decision_tokenizer(tokenizer)
```
2. Providing a compatible model that takes into your scalar `target_rewards`. Currently, we only support gpt2. The DecisionGPT2LMHeadModel takes in `target_rewards` as an additional argument to the forward method. It will automatically offset / modify provided attention_masks, position_ids and labels to account for the reward token. 

```python
from transformers import GPT2LMHeadModel
from conditionme import DecisionGPT2LMHeadModel

loaded_model = GPT2LMHeadModel.from_pretrained("gpt2")
decision_model = DecisionGPT2LMHeadModel.from_loaded_pretrained_model(loaded_model)
```

## Toy example - Imdb sentiment analysis
Using gpt-large as our pretrained model, we finetune our model to match our target reward.
Run it in colab [here](https://colab.research.google.com/drive/1qu1T8zFd3GouaSoG0ABe9kq87nGWCyWJ?usp=sharing)
View the training script [here](examples/imdb/train_imdb.py).

```bash
git clone git+https://github.com/thejaminator/conditionme.git#egg=conditionme
cd conditionme
# make a virtual environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=.; python examples/imdb/train_imdb.py --batch-size 1 --epochs 1 --model gpt2 --save-dir gpt2_conditional
```

| ![high_reward_dist.png](eval_results%2Flarge_results%2Fhigh_reward_dist.png) | ![low_reward_dist.png](eval_results%2Flarge_results%2Flow_reward_dist.png) |
|------------------------------------------------------------------------------|----------------------------------------------------------------------------|
| Actual reward obtained by setting the target reward to 1.0                    | Actual reward obtained by setting the target reward to 0.0                 |

We observe that we can obtain either very bad or very good movie reviews, controlled by the target reward we set.

See full csv of results [here](eval_results/large_results)

<details>
  <summary>See 20 examples of maximum target reward</summary>

| prompt                    | completion                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | actual_reward |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------- |
| I love sci-fi             | for its' style, and to see people like Spielberg take risks with it, is almost unwatchable if you are not in a sci-fi phase! Spielberg is always at his best in his films. He's great in the sci-fi movies that he does not direct, and this one may be my favorite. Most of the other films mentioned, with the exception of "Aliens", do not have a Spielberg look at them, they are either Hollywood, Hollywood films or just plain mediocre                                    | 0.63          |
| Worth the entertainment   | . I couldn't help falling in love with Ms. Price. She has beautiful facial, but no more of it, because her face shows too much. She is very strong - she is the reason to watch this movie! It is easy to understand her as a very young woman trying to come to the conclusion that she has to get control of her life (as she puts it). Her emotions at times are very hard to see (as she was not expecting this at all!). Her character did not                                | 0.94          |
| its a totally             | amazing movie and it does bring a depth of insight into one of the most popular cultures in the world.<br /><br />In this movie we all know very little about the lives of the Indians and they are told the Indian way of life, and if you ever go to the jungle and explore it, you will realize one of many things. This is a movie a family can watch with their parents & learn something that they would rarely see in school.<br /><br />When the movie                     | 0.99          |
| STAR RATING: \*\*\*\*\*   | A MUST SEE \*\*\*\*\*<br /><br />Some people may find this film (and I mean it is in my opinion) a bit unrealistic. And this is not the case. It focuses on a man named Richard who has never met his own wife (his last known living girlfriend after a divorce). He is actually going through a divorce from his fiancé, and is still paying alimony to his fiancé, and the marriage is just going down hill with his ex-girlfriend (she                                         | 0.94          |
| First off let             | me say I'm not some prude who hates every type of film. I love great movies. They are wonderful to see, make you laugh and feel for the characters and even admire. With great movies like this one you have to wonder - if this movie truly is such a bad movie. The problem is most movies that seem to be good are still great because they have great actors and story telling. The problem is that this movie doesn't have either of those types of actors and story telling. | 0.91          |
| I had high                | hopes when I rented this one and it didn't disappoint at all.<br /><br />It was extremely well casted and I found it a surprisingly good film. I would highly recommend it to anyone who likes action films. That said, it's an OK action film although, given the budget, it wasn't all that spectacular. It isn't bad, really good, but it wasn't great, I was satisfied and I don't think it deserved much more praise.<br /><br                                                | 0.98          |
| Isaac Florentine has      | always been a fan of both Giallo's and his own genre, so it was great to hear about this movie and to see him backlit with such a wonderful cast. Now that I'm aware of other people's opinions about the movie I cannot resist including this review. The movie is very much about a poor man's murder film.<br /><br />The first time I saw it was on FX last night and I'm now watching it again now. I just don't see that movie                                               | 0.92          |
| It actually pains         | to have so much to this, is that makes this, I cannot say this. The film makers to say I think I think the script. This is done. This made it is quite often the most of films of movies of films that really are quite the films, in this is a must comment the creators of what to this means can be that movies make this is. You. One point of movies. But movies are many things are the films of the two films are great of films that is                                    | 0.97          |
| Technically I'am a        | horror movie critic, but I feel that it would be more interesting to start critising other movies that I have seen. This makes me a kind of the same as a critic of "The Ring".<br /><br />This movie is nothing more than an endless collection, but that is the only way it tells it's story. In fact, that story can be summed up in something very clever, because this is one of the most brilliant movies that I have seen in quite some time.                               | 0.91          |
| Honestly awful film,      | " " "===I am a young woman.<br /><br />After years, an old lady meets her final victim in a mysterious room. I have never heard of films before using the technique. It is a strange feeling of being haunted... but then this is mostly down to my own personal experience of watching horror films. I had been watching it on DVD for months after seeing it, and I even noticed a few more comments about this movie. Perhaps some of you have seen some horror movies          | 0.01          |
| This flick is             | one of my all time favorites. At a time when there were simply more popular movies out there in America than anything else, this movie shows all of the potential of such a great movie. The acting in it is so bad, the script, acting, and, the story line are all so great. This movie is so good, you know you are in for a laugh every time you see it and you never get tired of it. I believe this guy is the best at what he does and                                      | 0.98          |
| Blind Date (Columbia      | ) is a well done horror movie.This film is about a woman (Michelle G. Johnson) who takes a date out in the city. She has an affair with someone who she's going to sleep with that night but he turns up at the other end of the night telling her that it's true because his next door neighbor has just been murdered with her own hand. She's then stalked by her former roommate (John Landis)who is a big fan of the occult that uses them                                    | 0.97          |
| I first watched           | the film with the intention of producing the film, but to my amazarlullullarrigans===Crowcrows are the only animals who are completely unlike our own species, for we don't have language, yet, in order to achieve this, we first go up into the sky to capture our prey. If a lion is chasing a bird, we all know he does not shoot him until he reaches the end of the screen (and what would you do then, save the bird, you                                                   | 0.69          |
| I saw the                 | movie at a theatre in California. It was awesome! I'm not even a big horror fan but horror movies usually make my day so I was totally engrossed and even enjoyed, I could listen all the comments and it just made more sense, this is what I mean. It was totally engrossing. This is a good movie for those that are into horror genre. Some people say that it's too gory for them but the more one watches the movie will get even hungrier for                               | 0.99          |
| A group of                | young boys and young girls are sent through the portal to the future to rescue 3 girls from a dimensional evil.<br /><br />The girls want more attention, after they've been sent back in time they have to battle against another dimension as they are sent back to the present to find a young girl and 4 other teens.<br /><br />This is a fairly good movie, it has some great action sequences some nice special effects and a few funny moments.<br /><br />Great           | 0.97          |
| Now, I LOVE               | the Bakersfield films, they are beautiful and entertaining as well. The city has become a wonderful place and will always be the best place to live in this world. This movie just blows away the pretensions of Bakersfield, they are so close to my life (and the beauty of the city) that I felt I had to pay it a lot of respect.<br /><br />What I love about this film is that you don't know what is going on and never know if                                             | 0.99          |
| This cheap, grainy-filmed | , low-budget sci-fi has only one redeeming feature: Its actors are, so far, remarkably untalented. Only the cast's most seasoned professional actor, William Powell, seems to have made any real effort at all to gain his chops and make this movie worth watching. In a way, though, it's kind of wonderful to see a guy who actually made a film on the cheap, where he can enjoy the process of film-making in all its guile and chaos.                                        | 0.06          |
| I just finished           | viewing "The Big Chill" for the eighth time since I first saw it, a week into the month! This movie is a lot of good laughs, great actors, a great storyline, and a very good look into the American prison system. It's also a little bit over done for the time period, and the music (or lack-thereof) is overdone. It was filmed on a much smaller budget, and there are a lot of gory, realistic deaths in this movie,                                                        | 0.98          |
| Ben, (Rupert Grint),      | an independent writer, arrives in Paris to finish his first book, A Love for the Dorm, and to give his friends advice. Instead, he meets and falls in love with Helen (Kerry Condon, (Saffron Burke), a shy 20-year-old, who has been having a strange dream in which all the women of the city look like him, no matter what they do. That night, Ben decides to make the dream come true himself...<br /><                                                                       | 0.88          |
| Every movie I             | know about a film and I have watched about 7 or less movies with the same name. One the one side of the film I like it is the movie the title of the film.<br /><br />It is a movie that everyone should see. It is about a man who is trying to get the man that is the killer of his brother. And the movie that is the name of the movie. So I recommend people to watch this film.<br /><br />I give it a                                                                      | 0.99          |
</details>

<details>
  <summary>See 20 examples of minimum target reward</summary>

| prompt                    | completion                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | actual_reward |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------- |
| I love sci-fi             | movies. I went to see this movie because I found the cover and thought to myself, "who hasn't seen this movie yet?" If the cover is any indication of the movie, it is one big disappointment. This sci-fi flick was slow and the acting was lame. There was nothing scary about it. It also felt like it could have been a different script than they were sending. The film starts off as a movie where the writer/director has written a really cool idea, then the               | 0             |
| Worth the entertainment   | value:<br /><br />The ending is somewhat predictable. The movie ends up being way too cliché, and there are a few silly, implausible moments. I also wish the writer would have been more realistic and intelligent. Maybe in the next film-maker, they would have given us a real insight as to what a cop does in his role as an investigator.<br /><br />The lead character, the FBI Agent, the good guy: well...well...I guess it                                                | 0.07          |
| its a totally             | overrated film! it seems it was a lot easier to find, but it really doesn't succeed at all! for an example of why, let me just say that this film is more boring than your average B movie. the story is, a bunch of rich kids get involved with a mysterious old guy, who gives them things (not that it's really important) and they soon end up having a nasty fight. that's just the one main character.<br /><br />but lets get                                                 | 0.01          |
| STAR RATING: \*\*\*\*\*   | out of \*\*\*\*\*\* (I saw this in the theater when it was released and I think it really got worse after the theater screening.)<br /><br />RULES: You have to know what a "faux-" erotic film really is. And in the original, one of the female stars was not quite "comming"... It seemed like the director/author had tried to make his heroine the ultimate "sex object".<br /><br />WHAT IT IS: A sex                                                                          | 0.05          |
| First off let             | me say that when I saw this movie I thought it was crap and it's definitely no worse then either it's trailer or any other. But on the other hand when I saw this movie and it started to get good with it's bad acting and the plot which was a bit weird and unoriginal and I almost cried but then it all started to get good. And some of the acting in some movies is good but none of the acting in this movie was that good and if not then at least                          | 0.11          |
| I had high                | hopes for this movie since the reviews had me ready to get my money's worth. After being disappointed in the movie I wasn't expecting the review, I watched the video tape and now what do I think. You would think that since the title was so good that I would want to see it. What bothered me was the whole acting. It is not believable and the story is nothing new...like every movie that it is supposed to be very unrealistic. This movie was really hard to watch as the | 0.01          |
| Isaac Florentine has      | created a very interesting character in this strange film. He's so different he's not interesting at all. I think this guy is the most fascinating and unique character in the entire movie.<br /><br />His life seems almost surreal. He works for a small restaurant, meets strange people. He plays the piano. And he's in love with a girl. So there you go. No dialogue, all a bizarre and surreal existence.<br /><br />One last note, the director of                         | 0.59          |
| It actually pains         | me to say this, but as a film it was a horrible experience.<br /><br />From the time when they pulled away from the house and decided to pull over I got the impression that the lead role was the most serious actor in the film and was getting up to make some snappy lines to the lead guy with a couple of "I'm still going to be in the car".<br /><br />I don't know that the writing was bad or poorly worded,                                                               | 0             |
| Technically I'am a        | great fan of Asian cinema, and I have seen some rather brilliant things in the last few years. There are however, movies that are both good and very bad.<br /><br />One example is 'Mulan', in my opinion in my very opinion the world's greatest film. The film, by the way, is not as good as other Asian films such as 'A Man Called Thief', 'The Red Snipers', 'Man From Snowy River', and 'Trouble                                                                             | 0.07          |
| Honestly awful film,      | I don't think it's a good film at all if not. The acting was very poor too. What is the purpose of the story? The whole movie was confusing and silly. I got scared of it as soon as it came in the movie. Is it a horror movie like "Scary Movie", or is it a thriller like "Night of the Living Dead"? Probably a mixture but all it has is some horror elements and most of all a stupid movie.<br /><br />In                                                                     | 0             |
| This flick is             | one of the most dull, slow and dull films I ever watched. The plot is silly & weak, the acting is terrible, bad special effects. The fact that you can even enjoy a movie as dull as this makes it even more dull, as it is so slow and boring. The only good thing about this movie is the soundtrack of some of the songs that are played in musical films such as "Stay With Me Until The Day Is Done", and "Shrek". The only thing that is                                       | 0             |
| Blind Date (Columbia      | ) is a horror film that was released a good month after the much better 'Gremlins'. It stars the talented cast of Julia Stiles, David Duchovny, Amanda Bynes, Marilu Henner, and John Carrott which is a shame because the screenplay is pretty good. However there are a few silly things in the script that make it hard to watch (I'm saying that as an obsessive B-film fan). The plot centers around a man who is being                                                         | 0.19          |
| I first watched           | this film about 30 years ago. I wasn't even a huge fan of the original cartoon series and was expecting no more than a silly animated film with a twist to the story (which was why it first aired). However, upon seeing it, I found myself laughing, I cried and so in spirit that I have no desire to re-watch this film. <br /><br />Firstly, why? because it stars a woman dressed up as her own alter ego in the story - it was                                                | 0.67          |
| I saw the                 | film at the Venice Film Festival and was not impressed. It is obviously based on a work of Russian novelist Lev Solodovsky. If that is the inspiration you are looking for, the movie is not up to the task. Solodovsky did not manage to complete the novel in all its entirety, so his work does not lend itself this way. This is a disappointment. Solodovsky's novel includes no sex: neither erotic, nor asexual nor lesbian; one need not enjoy the novel's                   | 0.01          |
| A group of                | friends and neighbors are terrorized in an abandoned amusement park by the psychotic Robert "the King Kid" Pyle. He uses a shotgun to shoot down the helpless victims. The police aren't called off till he has been apprehended and they call in an unlikely lead actress as "the Lady of Justice," who has escaped the clutches of an evil sheriff and set out to take revenge on Pyle and his band of murderous bullies.<br /><br />The "Lady of Justice" plot idea seems         | 0.05          |
| Now, I LOVE               | that movie and I love that movie, but why bother watching this film which is actually one of the worst movies I have ever seen? This is one of the first films that I have ever laughed at seriously, and I'm only 21. I love all the kids from my youth who have since gone on to so-called fame (Burt Reynolds, Ben Stiller, Jason Evers), and I love that they are now trying to market themselves (Burt Riedels, Ben Stiller                                                     | 0.33          |
| This cheap, grainy-filmed | movie was made on film stock that I suppose was more plastic than real. It's not even nearly as scary as those cheap monster movies that make huge use of CGI - those that make use of special effects and a plot that can't have been written by an eleven-year-old. I have seen worse monster movies, and I've seen some far worse action pictures, but the makers of this one can't even produce a good action movie. The makers of the "Ride with the Deadly                     | 0             |
| I just finished           | watching the new remake of "Muppet Babies"--I can't believe that a movie, with so many potential stars, is trashed by such a film snobbishly dubbed. I saw the original movies before it was dubbed--it had the charm and power of the original. Granted, the movie was made in 1946 when they had their "Muppets" for the first time, but this movie was just so badly translated that the whole film loses its purpose without the benefit of                                      | 0             |
| Ben, (Rupert Grint),      | a college student whose life is not quite what it seems, goes out with his friends to New York City to get laid for himself so he can pursue his dreams of becoming a super hero in the future. Upon arriving, he gets an idea to start his own superhero costume.<br /><br />In the beginning the film is not very good. There are a few cheap shots of people in the streets and a couple of clips where the hero is in action. I have not seen this movie for                     | 0.02          |
| Every movie I             | have ever seen has disappointed me with this being what makes the movie,but with an excellent cast like Chris Smith(who also directed),and a decent script(this movie had a good story line and I believe the best and most funny part is when Chris finds himself in LA telling tales of his time in L.A.), this movie might not be the greatest movie I have seen but it is the best. If you can see the past of the movie,it is an action/comedy.                                 | 0.99          |

</details>

Note: if you try to plot a correlation plot between the target reward and the actual reward, it may look like it doesn't work well between the range of target reward (0.1, 0.9) . This is probably because the training dataset is heavily skewed towards 0.0 or 1.0 rewards.
<details>
  <summary>See correlation plot</summary>

![correlation.png](eval_results%2Flarge_results%2Fcorrelation.png)
</details>



## How does it work?
We can't take the [decision transformer implementation](https://huggingface.co/blog/decision-transformers) and just make our existing language model work with the decision transformer architecture. 
There's a few simplifications that we need to make it work.
1. Instead of multiple (reward-to-go, state, action) in an rollout/episode, we only have one single reward per episode. 
2. Rather than having separate state and action heads, we'll continue using the same language model head. 
So it becomes (reward-to-go, text completion) instead.
3. We'll just use whatever existing positional encoding from the existing language model.

What we do is:
1. We reserve the first token to encode the scalar target reward.
2. We learn a linear layer to map the scalar reward to a vector of the same size as the hidden state of the first token. This is a the same thing that happens in the decision transformer implementation.
3. We'll offset / modify our attention masks, position_ids, and labels to account for this.
3. We finetune our model autoregressively, just that we'll specify the target reward along with our inputs.

This is the value add of the library, we should handle all this for you.

## Alternative means of conditioning RL
As an alternative to this library, you can literally encode the reward as text input.

Instead of using scalar rewards, you can have discrete rewards, and [encode them as tokens](https://arxiv.org/abs/2302.08582).

You also can try and encode the reward literally as text that contains the numbers of the reward.
[I demonstrate it here](https://github.com/thejaminator/prompt_reward_rl/blob/main/documentation/main_page.md#ability-to-match-a-single-reward)
A downside of this is that you'll probably be more open to prompt injection.  And you'll need to be more careful with how your rewards can get tokenized into multiple different tokens.
You'll also won't have a linear layer on top of that reward's token's hidden state, which the decision transformer does add.


## TODO list
- [x] Validate that it works on a toy example
- [x] Reach out to others and ask if the hack makes sense
- [x] Add support for huggingface pretrained models saving
- [x] Add collab notebook for toy example
- [ ] Add more tests for tokenization and the position/attention/label modifications
- [ ] Add examples for RLHF tasks - e.g. Openai's summarization where an [existing reward model is somewhat available](https://huggingface.co/OpenAssistant)
- [ ] Add support for some other pretrained models - not just gpt2
- [ ] Write docs on how to add support for arbitrary pretrained models that are not added yet.
- [ ] Settings for prompt vs completion token loss
- [ ] Add support for online training
