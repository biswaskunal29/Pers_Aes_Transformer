from wordcloud import WordCloud

import matplotlib.pyplot as plt
#%matplotlib inline

text = """2x army for vampires

heroes/lords
1x fighter
1x engineer


infantry (8)
center  2x irondrakes 2x longbeards
left	2x longbeards(great weapons) 
right	2x longbeards(great weapons) 

missile (3)
2x quareller
1x thunderer

artillery (3)
2x Cannon
1x Flame Cannon + engineer

gyro (4)
3x gyro copter(Brimstone)
1x gyro copter/bomber (anti infantry) 


carrier
2x irondrakes
2x longbeards
4x longbeards (great weapons)
1x flame cannon
3x gyro copter(Brimstone)
1x gyro copter/bomber (anti infantry) 



vamp bill

Heroes(4)
	2x fighter
	2x engineer

infantry(16)
	4x irondrakes
	4x longbeards
	8x longbeards(great weapons)

missile(6)
	4x quareller
	2x thunderer

artillery(6)
	4x cannon
	2x flame cannon

gyro(8)
	6x gyro copter(Brimstone)
	2x gyro copter/bomber (anti infantry)




3x army chaos

heroes/lords
2x fighter

infantry(9)
center 	3x longbeards(Great Weapons) + one of those fighter heroes
left	2x longbeards(Great Weapons)
right	2x longbeards(Great Weapons)
reserve	2x slayer(great weapons) 

missile(3)
3x thunderer

artillery(2)
2x Cannon 

gyro(4)
left	2x gyro copter(Brimstone)
right 	2x gyro copter(Brimstone) 




chaos bill

Heroes(6)
	6x fighter

infantry(27)
	21x longbeards(Great Weapons)
	6x  slayer
	
missile(9)
	9x thunderer

artillery(6)
	6x Cannon 

gyro(12)
	12x gyro copter(Brimstone)








Total Bill 

Heroes(10)
	5x fighter Lords
	3x fighter heroes
	2x engineer

infantry(43)
	4x  irondrakes (done)
	4x  longbeards (done)
	29x	longbeards(great weapons) (Done)
	6x  slayer (done)

missile(15)
	4x quareller (done)
	11x thunderer (done)

artillery(12)
	10x cannon (done)
	2x flame cannon (done)

gyro(20)
	18x gyro copter(Brimstone) (done)
	2x  gyro copter/bomber (anti infantry) (dones)


1x	north recruit 	12x artillery
	
3x	south recruit  	43x (25+8+6+4) infantry(- irondrakes) and quareller

2x	west recruit	35x (11+4+20) irondrakes and thunderer"""

#print(len(text))

wc = WordCloud(width = 800, height = 400).generate(text)
plt.imshow(wc)
plt.axis('off')



