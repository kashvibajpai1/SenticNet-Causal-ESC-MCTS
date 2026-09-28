# Qualitative examples (causal_mcts, seed=42)

## Example 1 (esconv-test-0)

**Context:**
- Seeker: hi i am okay, a little bit sad though
- Supporter: Okay. I am very sorry to hear that! Do you want to tell me more about that?
- Seeker: Well with the holidays coming up i have been very stressed and nervous about what i am going to do

**Strategy:** predicted=`Reflection` gold=`Affirmation` (mismatch)

**Generated reply:** It sounds like you're feeling overwhelmed by the stress of the holidays. It's great that you're seeking help now. Let's talk about how we can support each other through this period. What specifically would you like to discuss or share?

**Gold reply:** I really feel you there. Holidays are so hard.. especially with the way this year has been. Anything specific?

**ROUGE-L:** 0.127 | **BERTScore F1:** 0.735

## Example 2 (esconv-test-0)

**Context:**
- Seeker: Well with the holidays coming up i have been very stressed and nervous about what i am going to do
- Supporter: I really feel you there. Holidays are so hard.. especially with the way this year has been. Anything specific?
- Seeker: Yes. I am a college student and I live on campus. The day before I moved in my dad said some unforgivable things to me and we haven't spoke since
- Supporter: I have no choice but to go live at home over the break but I am very scared about it

**Strategy:** predicted=`Reflection` gold=`Question` (mismatch)

**Generated reply:** It's understandable to feel overwhelmed by the stress of moving into college and dealing with family dynamics during the holiday season. It can be helpful to take small steps towards adjusting your routine while maintaining a strong sense of community and support. Perhaps consider setting

**Gold reply:** Do you feel scared for your own self? Or of the situation in general? I'm sorry to hear that. Relationships with parents can be very difficult sometimes.

**ROUGE-L:** 0.135 | **BERTScore F1:** 0.749

## Example 3 (esconv-test-0)

**Context:**
- Seeker: I really feel you there. Holidays are so hard.. especially with the way this year has been. Anything specific?
- Supporter: Yes. I am a college student and I live on campus. The day before I moved in my dad said some unforgivable things to me and we haven't spoke since
- Seeker: I have no choice but to go live at home over the break but I am very scared about it
- Supporter: Do you feel scared for your own self? Or of the situation in general? I'm sorry to hear that. Relationships with parents can be very difficult sometimes.

**Strategy:** predicted=`Reflection` gold=`Self-disclosure` (mismatch)

**Generated reply:** It's understandable to feel overwhelmed by the stress of moving into a new environment like living at home during the holiday season. It's important to remember that everyone goes through these transitions differently, and it's completely normal to feel anxious or uncertain.

**Gold reply:** I'm around college age as well and also live at home. So I can understand your struggle.

**ROUGE-L:** 0.095 | **BERTScore F1:** 0.745

## Example 4 (esconv-test-0)

**Context:**
- Seeker: Do you feel scared for your own self? Or of the situation in general? I'm sorry to hear that. Relationships with parents can be very difficult sometimes.
- Supporter: I'm around college age as well and also live at home. So I can understand your struggle.
- Seeker: more for the situation in general. i know we will not speak but we will constantly be running into each other, my house is not big.
- Supporter: and i have no car to escape it

**Strategy:** predicted=`Reflection` gold=`Self-disclosure` (mismatch)

**Generated reply:** It's understandable how stressful and anxious you're feeling right now, especially given the holiday season. It sounds like you've been dealing with significant stressors both personally and professionally. Reflecting on these experiences might help you process them better. Remember

**Gold reply:** It's like we live the same life. I also have no car to escape! It seems so small, but it's such a huge stressor when you feel trapped in an environment you're not positive in. I feel your pain and I empathize with you completely.

**ROUGE-L:** 0.156 | **BERTScore F1:** 0.787

## Example 5 (esconv-test-0)

**Context:**
- Seeker: I'm around college age as well and also live at home. So I can understand your struggle.
- Supporter: more for the situation in general. i know we will not speak but we will constantly be running into each other, my house is not big.
- Seeker: and i have no car to escape it
- Supporter: It's like we live the same life. I also have no car to escape! It seems so small, but it's such a huge stressor when you feel trapped in an environment you're not positive in. I feel your pain and I empathize with you completely.

**Strategy:** predicted=`Reflection` gold=`Affirmation` (mismatch)

**Generated reply:** It sounds like you've navigated this challenging time quite well, despite the difficulties. Your feelings are valid; relationships can indeed be isolating and stressful. It's important to remember that everyone goes through tough times, and it's okay to

**Gold reply:** It'll be hard but I hope you can make it through it throughout the holidays and enjoy yourself some. Are there any pros to going back home? Any pets?

**ROUGE-L:** 0.197 | **BERTScore F1:** 0.721

## Example 6 (esconv-test-0)

**Context:**
- Seeker: It's like we live the same life. I also have no car to escape! It seems so small, but it's such a huge stressor when you feel trapped in an environment you're not positive in. I feel your pain and I empathize with you completely.
- Supporter: It'll be hard but I hope you can make it through it throughout the holidays and enjoy yourself some. Are there any pros to going back home? Any pets?
- Seeker: Thank you I appreciate that. I will be fine making it over the thanksgiving break but I am more nervous about covid-19 sending us home for good.
- Supporter: Not many to be honest. I have a hamster but he is at school with me so nothing at home to go back to

**Strategy:** predicted=`Affirmation` gold=`Restatement` (mismatch)

**Generated reply:** I'm here for you, Seeker. You've shared a lot, and I'm deeply sorry to hear about the challenges you're facing. Remember, it's okay to feel overwhelmed, and it's important to take care of yourself during

**Gold reply:** It sounds like Covid- 19 is going to be a personal stressor for you. It's such a strange thing to have to live with already, the pandemic, and i'm sorry that it might end up pushing you where you don't want to be.

**ROUGE-L:** 0.207 | **BERTScore F1:** 0.773

## Example 7 (esconv-test-0)

**Context:**
- Seeker: It'll be hard but I hope you can make it through it throughout the holidays and enjoy yourself some. Are there any pros to going back home? Any pets?
- Supporter: Thank you I appreciate that. I will be fine making it over the thanksgiving break but I am more nervous about covid-19 sending us home for good.
- Seeker: Not many to be honest. I have a hamster but he is at school with me so nothing at home to go back to
- Supporter: It sounds like Covid- 19 is going to be a personal stressor for you. It's such a strange thing to have to live with already, the pandemic, and i'm sorry that it might end up pushing you where you don't want to be.

**Strategy:** predicted=`Affirmation` gold=`Providing Suggestions` (mismatch)

**Generated reply:** I'm here for you, Seeker. You've shared a lot of emotions, and I'm deeply sorry for hearing that. Remember, it's okay to feel overwhelmed by these feelings. Just take deep breaths and remind yourself that you

**Gold reply:** Could you bring your hamster home with you? Even the smallest things could help a place feel more loving

**ROUGE-L:** 0.136 | **BERTScore F1:** 0.678

## Example 8 (esconv-test-0)

**Context:**
- Seeker: It sounds like Covid- 19 is going to be a personal stressor for you. It's such a strange thing to have to live with already, the pandemic, and i'm sorry that it might end up pushing you where you don't want to be.
- Supporter: Could you bring your hamster home with you? Even the smallest things could help a place feel more loving
- Seeker: Yes it is very strange and I know that it is a big stressor on all of us, i don't want to sound selfish.
- Supporter: Yes i am bringing him home with me so that is my little piece of joy that is coming along

**Strategy:** predicted=`Affirmation` gold=`Affirmation` (match)

**Generated reply:** I'm here for you, Seeker. Your struggles are valid, and I'm here to listen and offer support. Remember, it's okay to feel overwhelmed, but try to stay positive and focused on how you can still find joy and

**Gold reply:** I don't think you're selfish at all. You have feelings you are entitled to have and are expressing them which is healthy and I encourage that behavior!

**ROUGE-L:** 0.171 | **BERTScore F1:** 0.764

## Example 9 (esconv-test-0)

**Context:**
- Seeker: Could you bring your hamster home with you? Even the smallest things could help a place feel more loving
- Supporter: Yes it is very strange and I know that it is a big stressor on all of us, i don't want to sound selfish.
- Seeker: Yes i am bringing him home with me so that is my little piece of joy that is coming along
- Supporter: I don't think you're selfish at all. You have feelings you are entitled to have and are expressing them which is healthy and I encourage that behavior!

**Strategy:** predicted=`Affirmation` gold=`Providing Suggestions` (mismatch)

**Generated reply:** I'm here for you, Seeker. Your struggles are valid, and I'm here to listen and offer support. Remember, it's okay to feel overwhelmed by these emotions. Just take deep breaths and try to focus on the present

**Gold reply:** Good! I know how hard this situation can be. Very nerve wracking. The smallest things, even a hamster, can be so beneficial.

**ROUGE-L:** 0.065 | **BERTScore F1:** 0.712

## Example 10 (esconv-test-0)

**Context:**
- Seeker: Yes i am bringing him home with me so that is my little piece of joy that is coming along
- Supporter: I don't think you're selfish at all. You have feelings you are entitled to have and are expressing them which is healthy and I encourage that behavior!
- Seeker: Good! I know how hard this situation can be. Very nerve wracking. The smallest things, even a hamster, can be so beneficial.
- Supporter: Well thank you that does help me to feel better. Yes I am happy to bring him home with me and hoping I can have my niece over to intro duce her to him

**Strategy:** predicted=`Information` gold=`Reflection` (mismatch)

**Generated reply:** I'm here to listen and offer comfort during these challenging times. Your experience with the holiday season and the pandemic is quite common, and I'm sure you've found ways to cope. Remember, it's important to take care of yourself both

**Gold reply:** You sound like a very determined person in making this trip as stress free as possible. I commend you for that! Maybe you could have your niece over often to play with your hamster just to soften the stress of being there

**ROUGE-L:** 0.116 | **BERTScore F1:** 0.731

## Example 11 (esconv-test-0)

**Context:**
- Seeker: Good! I know how hard this situation can be. Very nerve wracking. The smallest things, even a hamster, can be so beneficial.
- Supporter: Well thank you that does help me to feel better. Yes I am happy to bring him home with me and hoping I can have my niece over to intro duce her to him
- Seeker: You sound like a very determined person in making this trip as stress free as possible. I commend you for that! Maybe you could have your niece over often to play with your hamster just to soften the stress of being there
- Supporter: It is surely my goal! I appreciate your kind words. I would love that if I had a car I could do that but I am not sure how much she will be able to come to my house. I am going to try though!

**Strategy:** predicted=`Information` gold=`Providing Suggestions` (mismatch)

**Generated reply:** I'm here to listen and offer comfort during these challenging times. Remember, it's okay to feel overwhelmed, and it's important to take care of yourself. Just knowing that you're not alone makes a huge difference. If you ever need

**Gold reply:** Is Uber an option for you? I know there's a lot of rideshare services but they're not always financially reasonable. But that could be a way to get away if you ever need too!

**ROUGE-L:** 0.253 | **BERTScore F1:** 0.744

## Example 12 (esconv-test-0)

**Context:**
- Seeker: You sound like a very determined person in making this trip as stress free as possible. I commend you for that! Maybe you could have your niece over often to play with your hamster just to soften the stress of being there
- Supporter: It is surely my goal! I appreciate your kind words. I would love that if I had a car I could do that but I am not sure how much she will be able to come to my house. I am going to try though!
- Seeker: Is Uber an option for you? I know there's a lot of rideshare services but they're not always financially reasonable. But that could be a way to get away if you ever need too!
- Supporter: that is very true i didn't even think of that to be honest! I may use that along the way, thank you for the reminder!

**Strategy:** predicted=`Information` gold=`Self-disclosure` (mismatch)

**Generated reply:** I'm here to listen and offer comfort during these challenging times. Remember, it's okay to feel overwhelmed, and it's important to take care of yourself. We're all in this together, and I believe in our ability to navigate this

**Gold reply:** Sometimes I forget little things like that to help myself whenever I get too involved in the emotional aspect of things. I'm glad I could suggest it to you!

**ROUGE-L:** 0.137 | **BERTScore F1:** 0.740

## Example 13 (esconv-test-0)

**Context:**
- Seeker: Is Uber an option for you? I know there's a lot of rideshare services but they're not always financially reasonable. But that could be a way to get away if you ever need too!
- Supporter: that is very true i didn't even think of that to be honest! I may use that along the way, thank you for the reminder!
- Seeker: Sometimes I forget little things like that to help myself whenever I get too involved in the emotional aspect of things. I'm glad I could suggest it to you!
- Supporter: Thank you! Well i appreciate your help a lot and hope you have a great day!

**Strategy:** predicted=`Information` gold=`Others` (mismatch)

**Generated reply:** I'm here to listen and offer comfort during these challenging times. Your story is incredibly relatable, and I'm truly sorry to hear about the stress and anxiety you're experiencing. Remember, it's okay to feel overwhelmed; it's important

**Gold reply:** Of course! I hope you do as well and happy holidays!

**ROUGE-L:** 0.075 | **BERTScore F1:** 0.693

## Example 14 (esconv-test-1)

**Context:**
- Seeker: Hai. My name is Tom and i really need your help.
- Supporter: ok tell me
- Seeker: I am really sad and stressed up knowing that I am soon losing my job due to the effects of corona virus. For sure if I lose my job now I might go to depression because I have a family to take care of.

**Strategy:** predicted=`Reflection` gold=`Affirmation` (mismatch)

**Generated reply:** It's great to hear about your situation, Tom. Losing a job during such an uncertain time can be incredibly challenging. It's important to remember that you're not alone in this; many people are facing similar challenges. If you need anything

**Gold reply:** ok, don't worry friend,

**ROUGE-L:** 0.000 | **BERTScore F1:** 0.658

## Example 15 (esconv-test-1)

**Context:**
- Seeker: ok tell me
- Supporter: I am really sad and stressed up knowing that I am soon losing my job due to the effects of corona virus. For sure if I lose my job now I might go to depression because I have a family to take care of.
- Seeker: ok, don't worry friend,
- Supporter: Are there any ways you know that could help me convince my boss that that it is not the right time to cut down on manpower?

**Strategy:** predicted=`Reflection` gold=`Affirmation` (mismatch)

**Generated reply:** It's important to approach this situation with empathy and understanding. It sounds like you're feeling overwhelmed by the stress of job loss and financial strain. While it's understandable to feel depressed, it's crucial to maintain a positive outlook. Consider reaching

**Gold reply:** lot of people lose the job in this corona time

**ROUGE-L:** 0.074 | **BERTScore F1:** 0.694

## Example 16 (esconv-test-1)

**Context:**
- Seeker: I am really sad and stressed up knowing that I am soon losing my job due to the effects of corona virus. For sure if I lose my job now I might go to depression because I have a family to take care of.
- Supporter: ok, don't worry friend,
- Seeker: Are there any ways you know that could help me convince my boss that that it is not the right time to cut down on manpower?
- Supporter: lot of people lose the job in this corona time

**Strategy:** predicted=`Reflection` gold=`Affirmation` (mismatch)

**Generated reply:** It's understandable to feel overwhelmed by the situation. It can be helpful to remind yourself that many companies are facing similar challenges and may be looking for ways to manage their workforce effectively. Consider reaching out to HR or human resources to discuss potential solutions

**Gold reply:** just explain you situation.

**ROUGE-L:** 0.040 | **BERTScore F1:** 0.717

## Example 17 (esconv-test-1)

**Context:**
- Seeker: lot of people lose the job in this corona time
- Supporter: just explain you situation.
- Seeker: Incase i lose the job, are there ways that can help me stay out of stress of losing the job?
- Supporter: I am worried that losing a job may cause me a lot of harm since my bills depend on my current job.

**Strategy:** predicted=`Reflection` gold=`Affirmation` (mismatch)

**Generated reply:** It's important to remember that everyone experiences different situations differently, and what works for one person might not work for another. It's crucial to communicate openly with your boss about your concerns and feelings. They may be able to offer some guidance or

**Gold reply:** definitely, you have to try to get the other work

**ROUGE-L:** 0.073 | **BERTScore F1:** 0.717

## Example 18 (esconv-test-1)

**Context:**
- Seeker: just explain you situation.
- Supporter: Incase i lose the job, are there ways that can help me stay out of stress of losing the job?
- Seeker: I am worried that losing a job may cause me a lot of harm since my bills depend on my current job.
- Supporter: definitely, you have to try to get the other work

**Strategy:** predicted=`Reflection` gold=`Affirmation` (mismatch)

**Generated reply:** It's important to remember that everyone faces challenges differently, and it's okay to seek advice from others who understand what you're going through. Your feelings about losing your job are valid, and it's crucial to prioritize your mental health during these

**Gold reply:** don't give up your confident,

**ROUGE-L:** 0.039 | **BERTScore F1:** 0.714
