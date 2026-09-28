# Qualitative examples (causal_mcts, seed=42)

## Example 1 (esconv-test-0)

**Context:**
- Seeker: hi i am okay, a little bit sad though
- Supporter: Okay. I am very sorry to hear that! Do you want to tell me more about that?
- Seeker: Well with the holidays coming up i have been very stressed and nervous about what i am going to do

**Strategy:** predicted=`Question` gold=`Affirmation` (mismatch)

**Generated reply:** It sounds like you're feeling overwhelmed by the stress of the holidays. Have you considered taking some time off or seeking professional help? It's important to take care of your mental health during this time.

**Gold reply:** I really feel you there. Holidays are so hard.. especially with the way this year has been. Anything specific?

**ROUGE-L:** 0.107 | **BERTScore F1:** 0.728

## Example 2 (esconv-test-0)

**Context:**
- Seeker: Well with the holidays coming up i have been very stressed and nervous about what i am going to do
- Supporter: I really feel you there. Holidays are so hard.. especially with the way this year has been. Anything specific?
- Seeker: Yes. I am a college student and I live on campus. The day before I moved in my dad said some unforgivable things to me and we haven't spoke since
- Supporter: I have no choice but to go live at home over the break but I am very scared about it

**Strategy:** predicted=`Question` gold=`Question` (match)

**Generated reply:** It's understandable how stressful moving to live at home can be for someone like you. It's important to take care of yourself during this time. Have you considered reaching out to your friends or family for support? They might offer their friendship as

**Gold reply:** Do you feel scared for your own self? Or of the situation in general? I'm sorry to hear that. Relationships with parents can be very difficult sometimes.

**ROUGE-L:** 0.082 | **BERTScore F1:** 0.760

## Example 3 (esconv-test-0)

**Context:**
- Seeker: I really feel you there. Holidays are so hard.. especially with the way this year has been. Anything specific?
- Supporter: Yes. I am a college student and I live on campus. The day before I moved in my dad said some unforgivable things to me and we haven't spoke since
- Seeker: I have no choice but to go live at home over the break but I am very scared about it
- Supporter: Do you feel scared for your own self? Or of the situation in general? I'm sorry to hear that. Relationships with parents can be very difficult sometimes.

**Strategy:** predicted=`Question` gold=`Self-disclosure` (mismatch)

**Generated reply:** It's understandable to feel overwhelmed by such circumstances. It sounds like you're dealing with significant stress and uncertainty. Remember, it's okay to seek help or talk through these feelings. You don't need to keep everything bottled up; just reach

**Gold reply:** I'm around college age as well and also live at home. So I can understand your struggle.

**ROUGE-L:** 0.033 | **BERTScore F1:** 0.729

## Example 4 (esconv-test-0)

**Context:**
- Seeker: Do you feel scared for your own self? Or of the situation in general? I'm sorry to hear that. Relationships with parents can be very difficult sometimes.
- Supporter: I'm around college age as well and also live at home. So I can understand your struggle.
- Seeker: more for the situation in general. i know we will not speak but we will constantly be running into each other, my house is not big.
- Supporter: and i have no car to escape it

**Strategy:** predicted=`Question` gold=`Self-disclosure` (mismatch)

**Generated reply:** It sounds like you're feeling a lot of stress and anxiety right now. It's understandable to feel overwhelmed by all the responsibilities and uncertainties associated with moving to college and living alone. However, it's important to remember that you're not alone

**Gold reply:** It's like we live the same life. I also have no car to escape! It seems so small, but it's such a huge stressor when you feel trapped in an environment you're not positive in. I feel your pain and I empathize with you completely.

**ROUGE-L:** 0.194 | **BERTScore F1:** 0.799

## Example 5 (esconv-test-0)

**Context:**
- Seeker: I'm around college age as well and also live at home. So I can understand your struggle.
- Supporter: more for the situation in general. i know we will not speak but we will constantly be running into each other, my house is not big.
- Seeker: and i have no car to escape it
- Supporter: It's like we live the same life. I also have no car to escape! It seems so small, but it's such a huge stressor when you feel trapped in an environment you're not positive in. I feel your pain and I empathize with you completely.

**Strategy:** predicted=`Question` gold=`Affirmation` (mismatch)

**Generated reply:** I understand how much you're feeling. It must be tough to navigate through these challenging times. Let's talk about how you might start to find ways to cope or adjust. What strategies do you think could help you manage your stress better during

**Gold reply:** It'll be hard but I hope you can make it through it throughout the holidays and enjoy yourself some. Are there any pros to going back home? Any pets?

**ROUGE-L:** 0.133 | **BERTScore F1:** 0.737

## Example 6 (esconv-test-0)

**Context:**
- Seeker: It's like we live the same life. I also have no car to escape! It seems so small, but it's such a huge stressor when you feel trapped in an environment you're not positive in. I feel your pain and I empathize with you completely.
- Supporter: It'll be hard but I hope you can make it through it throughout the holidays and enjoy yourself some. Are there any pros to going back home? Any pets?
- Seeker: Thank you I appreciate that. I will be fine making it over the thanksgiving break but I am more nervous about covid-19 sending us home for good.
- Supporter: Not many to be honest. I have a hamster but he is at school with me so nothing at home to go back to

**Strategy:** predicted=`Reflection` gold=`Restatement` (mismatch)

**Generated reply:** It sounds like you've navigated the holiday season with considerable stress and anxiety. Reflecting on your experience, it's clear how challenging it can be to navigate relationships and responsibilities during these times. Your feelings of being trapped and unable to communicate

**Gold reply:** It sounds like Covid- 19 is going to be a personal stressor for you. It's such a strange thing to have to live with already, the pandemic, and i'm sorry that it might end up pushing you where you don't want to be.

**ROUGE-L:** 0.202 | **BERTScore F1:** 0.770

## Example 7 (esconv-test-0)

**Context:**
- Seeker: It'll be hard but I hope you can make it through it throughout the holidays and enjoy yourself some. Are there any pros to going back home? Any pets?
- Supporter: Thank you I appreciate that. I will be fine making it over the thanksgiving break but I am more nervous about covid-19 sending us home for good.
- Seeker: Not many to be honest. I have a hamster but he is at school with me so nothing at home to go back to
- Supporter: It sounds like Covid- 19 is going to be a personal stressor for you. It's such a strange thing to have to live with already, the pandemic, and i'm sorry that it might end up pushing you where you don't want to be.

**Strategy:** predicted=`Reflection` gold=`Providing Suggestions` (mismatch)

**Generated reply:** It's important to remember that everyone copes differently with stress and change. While navigating the holiday season can be challenging, it's crucial to prioritize your mental health and seek support from friends or family who understand how you're feeling. Remember,

**Gold reply:** Could you bring your hamster home with you? Even the smallest things could help a place feel more loving

**ROUGE-L:** 0.097 | **BERTScore F1:** 0.703

## Example 8 (esconv-test-0)

**Context:**
- Seeker: It sounds like Covid- 19 is going to be a personal stressor for you. It's such a strange thing to have to live with already, the pandemic, and i'm sorry that it might end up pushing you where you don't want to be.
- Supporter: Could you bring your hamster home with you? Even the smallest things could help a place feel more loving
- Seeker: Yes it is very strange and I know that it is a big stressor on all of us, i don't want to sound selfish.
- Supporter: Yes i am bringing him home with me so that is my little piece of joy that is coming along

**Strategy:** predicted=`Reflection` gold=`Affirmation` (mismatch)

**Generated reply:** It's important to remember that everyone goes through these challenges differently, and it's okay to feel overwhelmed or anxious. Just knowing that you're not alone and that others are experiencing similar feelings can provide comfort during this time. Remember, you're

**Gold reply:** I don't think you're selfish at all. You have feelings you are entitled to have and are expressing them which is healthy and I encourage that behavior!

**ROUGE-L:** 0.110 | **BERTScore F1:** 0.760

## Example 9 (esconv-test-0)

**Context:**
- Seeker: Could you bring your hamster home with you? Even the smallest things could help a place feel more loving
- Supporter: Yes it is very strange and I know that it is a big stressor on all of us, i don't want to sound selfish.
- Seeker: Yes i am bringing him home with me so that is my little piece of joy that is coming along
- Supporter: I don't think you're selfish at all. You have feelings you are entitled to have and are expressing them which is healthy and I encourage that behavior!

**Strategy:** predicted=`Reflection` gold=`Providing Suggestions` (mismatch)

**Generated reply:** It's important to remember that everyone copes with stress differently, and it's perfectly normal to feel overwhelmed or anxious during these times. Just knowing that you're not alone in your struggles can provide much-needed comfort and understanding. Remember, you

**Gold reply:** Good! I know how hard this situation can be. Very nerve wracking. The smallest things, even a hamster, can be so beneficial.

**ROUGE-L:** 0.062 | **BERTScore F1:** 0.728

## Example 10 (esconv-test-0)

**Context:**
- Seeker: Yes i am bringing him home with me so that is my little piece of joy that is coming along
- Supporter: I don't think you're selfish at all. You have feelings you are entitled to have and are expressing them which is healthy and I encourage that behavior!
- Seeker: Good! I know how hard this situation can be. Very nerve wracking. The smallest things, even a hamster, can be so beneficial.
- Supporter: Well thank you that does help me to feel better. Yes I am happy to bring him home with me and hoping I can have my niece over to intro duce her to him

**Strategy:** predicted=`Affirmation` gold=`Reflection` (mismatch)

**Generated reply:** I'm here for you, Seeker. Your struggles are valid, and I'm here to listen and offer comfort. Remember, it's okay to feel overwhelmed, but take care of yourself first. Let's focus on finding ways to cope

**Gold reply:** You sound like a very determined person in making this trip as stress free as possible. I commend you for that! Maybe you could have your niece over often to play with your hamster just to soften the stress of being there

**ROUGE-L:** 0.171 | **BERTScore F1:** 0.707

## Example 11 (esconv-test-0)

**Context:**
- Seeker: Good! I know how hard this situation can be. Very nerve wracking. The smallest things, even a hamster, can be so beneficial.
- Supporter: Well thank you that does help me to feel better. Yes I am happy to bring him home with me and hoping I can have my niece over to intro duce her to him
- Seeker: You sound like a very determined person in making this trip as stress free as possible. I commend you for that! Maybe you could have your niece over often to play with your hamster just to soften the stress of being there
- Supporter: It is surely my goal! I appreciate your kind words. I would love that if I had a car I could do that but I am not sure how much she will be able to come to my house. I am going to try though!

**Strategy:** predicted=`Affirmation` gold=`Providing Suggestions` (mismatch)

**Generated reply:** I'm here for you, Seeker. Your struggles are valid, and I'm here to listen and offer comfort. Remember, it's okay to feel overwhelmed, but take care of yourself first. Let's focus on finding ways to cope

**Gold reply:** Is Uber an option for you? I know there's a lot of rideshare services but they're not always financially reasonable. But that could be a way to get away if you ever need too!

**ROUGE-L:** 0.184 | **BERTScore F1:** 0.724

## Example 12 (esconv-test-0)

**Context:**
- Seeker: You sound like a very determined person in making this trip as stress free as possible. I commend you for that! Maybe you could have your niece over often to play with your hamster just to soften the stress of being there
- Supporter: It is surely my goal! I appreciate your kind words. I would love that if I had a car I could do that but I am not sure how much she will be able to come to my house. I am going to try though!
- Seeker: Is Uber an option for you? I know there's a lot of rideshare services but they're not always financially reasonable. But that could be a way to get away if you ever need too!
- Supporter: that is very true i didn't even think of that to be honest! I may use that along the way, thank you for the reminder!

**Strategy:** predicted=`Affirmation` gold=`Self-disclosure` (mismatch)

**Generated reply:** I'm here for you, Seeker. Your struggles are valid, and I'm here to listen and offer comfort. Remember, it's okay to feel overwhelmed by these emotions. Just take deep breaths and remember that you're strong enough

**Gold reply:** Sometimes I forget little things like that to help myself whenever I get too involved in the emotional aspect of things. I'm glad I could suggest it to you!

**ROUGE-L:** 0.171 | **BERTScore F1:** 0.730

## Example 13 (esconv-test-0)

**Context:**
- Seeker: Is Uber an option for you? I know there's a lot of rideshare services but they're not always financially reasonable. But that could be a way to get away if you ever need too!
- Supporter: that is very true i didn't even think of that to be honest! I may use that along the way, thank you for the reminder!
- Seeker: Sometimes I forget little things like that to help myself whenever I get too involved in the emotional aspect of things. I'm glad I could suggest it to you!
- Supporter: Thank you! Well i appreciate your help a lot and hope you have a great day!

**Strategy:** predicted=`Affirmation` gold=`Others` (mismatch)

**Generated reply:** I'm here for you, Seeker. Your struggles are valid, and I'm here to listen and offer support. Remember, it's okay to feel overwhelmed, and I'm here to help guide you through these challenging times. Let's

**Gold reply:** Of course! I hope you do as well and happy holidays!

**ROUGE-L:** 0.118 | **BERTScore F1:** 0.680

## Example 14 (esconv-test-1)

**Context:**
- Seeker: Hai. My name is Tom and i really need your help.
- Supporter: ok tell me
- Seeker: I am really sad and stressed up knowing that I am soon losing my job due to the effects of corona virus. For sure if I lose my job now I might go to depression because I have a family to take care of.

**Strategy:** predicted=`Question` gold=`Affirmation` (mismatch)

**Generated reply:** It sounds like you're going through a tough time, Tom. Losing a job can be incredibly challenging, especially when it comes with the added stress of caring for a loved one. It's important to remember that you don't have to handle

**Gold reply:** ok, don't worry friend,

**ROUGE-L:** 0.082 | **BERTScore F1:** 0.662

## Example 15 (esconv-test-1)

**Context:**
- Seeker: ok tell me
- Supporter: I am really sad and stressed up knowing that I am soon losing my job due to the effects of corona virus. For sure if I lose my job now I might go to depression because I have a family to take care of.
- Seeker: ok, don't worry friend,
- Supporter: Are there any ways you know that could help me convince my boss that that it is not the right time to cut down on manpower?

**Strategy:** predicted=`Question` gold=`Affirmation` (mismatch)

**Generated reply:** It's important to approach this situation with empathy and understanding. Losing a job can be incredibly challenging, but it's crucial to maintain your mental health as well. Consider reaching out to HR or a human resources representative for guidance on how to handle

**Gold reply:** lot of people lose the job in this corona time

**ROUGE-L:** 0.073 | **BERTScore F1:** 0.688

## Example 16 (esconv-test-1)

**Context:**
- Seeker: I am really sad and stressed up knowing that I am soon losing my job due to the effects of corona virus. For sure if I lose my job now I might go to depression because I have a family to take care of.
- Supporter: ok, don't worry friend,
- Seeker: Are there any ways you know that could help me convince my boss that that it is not the right time to cut down on manpower?
- Supporter: lot of people lose the job in this corona time

**Strategy:** predicted=`Question` gold=`Affirmation` (mismatch)

**Generated reply:** It's understandable to feel overwhelmed by the situation. Losing a job during such an uncertain period can be incredibly stressful. It's important to remember that everyone copes with stress differently. If you're feeling particularly anxious or depressed, consider reaching out

**Gold reply:** just explain you situation.

**ROUGE-L:** 0.043 | **BERTScore F1:** 0.692

## Example 17 (esconv-test-1)

**Context:**
- Seeker: lot of people lose the job in this corona time
- Supporter: just explain you situation.
- Seeker: Incase i lose the job, are there ways that can help me stay out of stress of losing the job?
- Supporter: I am worried that losing a job may cause me a lot of harm since my bills depend on my current job.

**Strategy:** predicted=`Question` gold=`Affirmation` (mismatch)

**Generated reply:** It's important to remember that everyone experiences different situations differently. Losing a job during a pandemic like the one we're currently facing can be challenging but also incredibly valuable for personal growth and resilience. It's crucial to communicate openly with your employer about

**Gold reply:** definitely, you have to try to get the other work

**ROUGE-L:** 0.071 | **BERTScore F1:** 0.695

## Example 18 (esconv-test-1)

**Context:**
- Seeker: just explain you situation.
- Supporter: Incase i lose the job, are there ways that can help me stay out of stress of losing the job?
- Seeker: I am worried that losing a job may cause me a lot of harm since my bills depend on my current job.
- Supporter: definitely, you have to try to get the other work

**Strategy:** predicted=`Question` gold=`Affirmation` (mismatch)

**Generated reply:** I'm sorry to hear about your situation. Losing a job during such an uncertain period can indeed be challenging, but it's important to remember that many people find new opportunities or adjust their roles even when they're struggling. It's okay to

**Gold reply:** don't give up your confident,

**ROUGE-L:** 0.039 | **BERTScore F1:** 0.694
