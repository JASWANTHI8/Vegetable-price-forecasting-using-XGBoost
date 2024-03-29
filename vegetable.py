def crop(crop_name):
    crop_data = {
        "Banana": ["/static/images/banana.jpg", "Raw bananas are rich in potassium, which works as a vasodilator and controls blood pressure levels. It also prevents many heart conditions, such as atherosclerosis and heart attack, and improves heart health. Green bananas have a low glycemic index and slowly release the insulin hormone after consumption.Green bananas have a high amount of dietary fibers and resistant starch, which aid digestion, keep us full for a long time, and help in weight management. They are also beneficial in various stomach ailments, such as gastric ulcers, bloating, constipation, diarrhea, and bacterial infection of the digestive tract.In addition to minerals, green bananas are rich in various vitamins, including vitamins B6 and C. Vitamin B6 helps in numerous enzymatic processes in our body and boosts our metabolism.Green bananas maintain electrolyte balance in our bodies."],
        "Beetroot": ["/static/images/beetroot.jpg","Beetroots are a good source of nutrients, fiber, and many plant compounds. The health benefits of this vegetable include improved heart health, the ability to reduce blood pressure, and enhanced exercise capacity.Beetroot (Beta vulgaris) is a root vegetable also known as red beet, table beet, garden beet, or just beet.Packed with essential nutrients, beetroots are a great source of fiber, folate (vitamin B9), manganese, potassium, iron, and vitamin C.Raw or cooked beetroot offers about 8–10% carbs.Simple sugars — such as glucose and fructose — make up 70% and 80% of the carbs in raw and cooked beetroots, respectively.Beetroots are high in fiber, providing about 2–3 grams in each 3/4-cup (100-gram) raw serving.Beetroots have a glycemic index (GI) score of 61, which is considered medium."],
        "Bittergourd": ["/static/images/bittergourd.jpg","Bitter melon is high in vitamins A and C and other nutrients. It contains compounds that may have health benefits. But it may cause some side effects.Bitter melon — also known as bitter gourd or Momordica charantia — is a tropical vine that belongs to the gourd family and is closely related to zucchini, squash, pumpkin, and cucumber.It’s cultivated around the world for its edible fruit, which is considered a staple in many types of Asian cuisine.Bitter melon is especially rich in vitamin C, an important micronutrient involved in disease prevention, bone formation, and wound healing.It’s also high in vitamin A, a fat-soluble vitamin that promotes skin health and proper vision.It provides folate, which is essential for growth and development, as well as smaller amounts of potassium, zinc, and iron."],
        "Bottlegourd": ["/static/images/bottlegourd.jpg","Bottle gourd has been used traditionally to help with many health conditions like fever, cough, pain, and asthma. It has been used since ancient times for its benefits. It is also considered a good source of vitamin B, C, and other nutrients. It is known for its shape, a bottle, dumbbell, or oval shape.Bottle gourd might have properties that may be good for the liver, as per several animal studies. Bottle gourd may offer many potential benefits, helpful in liver condition and functions.Consuming bottle gourd may show beneficial effects on the memory. Certain compounds in bottle gourd may show pain-relieving and central nervous system (CNS) depressant activity by acting on the brain.Bottle gourd (Lauki) extract might act against the cancerous cells, as per an animal study. "],
        "Brinjal": ["/static/images/brinjal.jpg","Brinjal, also known as eggplant, is a nutrient-dense vegetable that is rich in several essential vitamins and minerals. It is a good source of fiber, potassium, vitamin C, and vitamin B6. These nutrients are essential for maintaining good health and can help prevent various diseases.Brinjal is a rich source of antioxidants, including anthocyanins and chlorogenic acid. These antioxidants are essential for maintaining good health and protecting the body from oxidative stress and free radical damage. Anthocyanins have been shown to have anti-inflammatory properties, while chlorogenic acid has been linked to a reduced risk of certain types of cancer.Brinjal contains compounds that can help lower blood pressure and cholesterol levels, which are two significant risk factors for heart disease."],
        "Cabbage": ["/static/images/cabbage.jpg","Cabbage is highly nutritious and rich in vitamin C, fiber, and vitamin K. Some research suggests that it may have health benefits that include supporting digestion and heart health, among others.Cabbage also contains small amounts of other micronutrients, including vitamin A, iron, and riboflavinIn addition, cabbage is high in fiber and contains powerful antioxidants, including polyphenols and sulfur compoundsAntioxidants protect the body from damage caused by free radicals. Free radicals are molecules that have an odd number of electrons, making them unstable. When their levels become too high, they can damage your cells.Cabbage is especially high in vitamin C, a potent antioxidant that may protect against heart disease, certain cancers, and vision loss "],
        "Capsicum": ["/static/images/capsicum.jpg","Bell peppers (Capsicum annuum) are fruits that belong to the nightshade family. They are low in calories and rich in vitamin C and other antioxidants, making them an excellent addition to a healthy diet.Bell peppers come in various colors, such as red, yellow, orange, and green — which are unripe.Bell peppers are primarily composed of carbs, which account for most of their calorie content — with 3.5 ounces (100 grams) holding 6 grams of carbs.Bell peppers are mainly made up of water and carbs. Most of the carbs are sugars, such as glucose and fructose. Bell peppers are also a decent source of fiber.Bell peppers are very high in vitamin C, with a single one providing up to 169% of the RDI. Other vitamins and minerals in bell peppers include vitamin K1, vitamin E, vitamin A, folate, and potassium."],
        "Carrot": ["/static/images/carrot.jpg","Carrots contain many nutrients, including beta carotene and antioxidants, that may support your overall health as part of a nutrient-rich diet.It is crunchy, tasty, and highly nutritious. Carrots are a particularly good source of beta carotene, fiber, vitamin K1, potassium, and antioxidants.Orange carrots get their bright color from beta carotene, an antioxidant that your body converts into vitamin A.Carrots are mainly composed of water and carbs.Soluble fibers can lower blood sugar levels by slowing down your digestion of sugar and starch.Soluble fibers can lower blood sugar levels by slowing down your digestion of sugar and starch.Carrots are about 10% carbs, consisting of starch, fiber, and simple sugars. They are extremely low in fat and protein.Carrots are a good source of several vitamins and minerals."],
        "Cassava": ["/static/images/cassava.jpg","Cassava is a root vegetable that contains vitamin C and copper. It may also contain harmful compounds if consumed raw.Cassava is a root vegetable widely consumed in many countries around the globe.It provides many important nutrients, including resistant starch, which may have health benefits.Nevertheless, as with all foods, you should be mindful to consume it in moderation. This is especially true considering that it’s fairly high in calories and contains potentially harmful chemicals.Cassava is a versatile root vegetable that’s widely consumed in several parts of the world. It’s also what tapioca starch is made from. You must cook it before eating it, as the raw form can be poisonous.Cassava is a significant source of carbs. It also provides a little fiber, vitamins, and minerals."],
        "Cauliflower": ["/static/images/cauliflower.jpg", "Cauliflower contains many nutrients and plant compounds that may reduce the risk of several diseases, including heart disease and cancer.Cauliflower is an excellent source of vitamins and minerals, including vitamin C, folate, and vitamin K.Cauliflower contains a high amount of fiber, which is important for digestive health and may reduce the risk of several chronic diseases.Many of cauliflower’s nutrients act as antioxidants, which are the substances that help protect our bodies from cell damage linked to diseases such as cancer.Cauliflower also contains choline, roughly 10% of the daily goal per cup, per the USDA.The fiber in cauliflower—nearly 12 grams per medium head, per the USDA—supports digestive health, promotes bowel regularity, and feeds beneficial bacteria in the gut tied to anti-inflammation, immunity, and mood."],
        "Clusterbeans": ["/static/images/clusterbeans.jpg","Cluster beans or Gavar is classified as a humble vegetable. It was found in the wilds but since time has come to be found as edible food. They are also used for cultivation. It grows in semiarid areas with regular and frequent rainfall. It has been in existence for many centuries but the sad part is that not many know the health benefits of it.Surprisingly cluster beans have all the health benefits which attribute to a good and healthy human body. Cluster Beans can help in strengthening bones and keep your heart healthy by managing the pumping of blood. This leads to having a controlled blood pressure condition etc. With the hike in percentages of contracting to the deadly disease of diabetes all around the world, cluster bean is one vegetable that is excellent in controlling blood sugar levels."],
        "Coconut": ["/static/images/coconut.jpg","Coconut is rich in fiber and MCTs (medium-chain triglycerides). Consuming it may offer a number of benefits, including improved digestion, heart health, and weight loss.Coconuts are the large seeds of coconut palms (Cocos nucifera), which grow in tropical climates. Their brown, fibrous husks conceal the meat inside.Coconut is a unique fruit because of its high fat content. Around 89% of the fat in its meat is saturated.Since these fruits are likewise high in fat, they can help your body absorb fat-soluble nutrients, including vitamins A, D, E, and K.Most of this fiber is insoluble, meaning that it doesn’t get digested. Instead, it works to move food through your digestive system and aids bowel health."],
        "Coriander": ["/static/images/coriander.jpg","Coriander is a fragrant, antioxidant-rich herb that has many culinary uses and health benefits. It can help lower your blood sugars, fight infections, and promote heart, brain, skin, and digestive health.Animal studies suggest that coriander seeds reduce blood sugar by promoting enzyme activity that helps remove sugar from the blood.Its antioxidants have been shown to fight inflammation in your body.Coriander extract appears to act as a diuretic, helping your body flush excess sodium and water. This may lower your blood pressure.Coriander may protect your heart by lowering blood pressure and LDL (bad) cholesterol while increasing HDL (good) cholesterol. A spice-rich diet appears to be associated with a lower risk of heart disease."],
        "Cucumber":  ["/static/images/cucumber.jpg","Cucumber is a nutritious fruit with a high water content. Eating cucumber may help lower blood sugar, prevent constipation, and support weight loss.Cucumbers are low in calories but high in water and several important vitamins and minerals. Eating cucumbers with the peel provides the maximum amount of nutrients.Cucumbers contain antioxidants, including flavonoids and tannins, which prevent the accumulation of harmful free radicals and may reduce the risk of chronic disease.Cucumbers are composed of about 96% water, which may increase hydration and help you meet your daily fluid needs.Cucumbers are low in calories, high in water and can be used as a low-calorie topping for many dishes. All of these may aid in weight loss."],
        "Drumstick": ["/static/images/drumstick.jpg","Drumstick comes with a nutrient punch such as vitamins A, C, K, B and minerals -iron, calcium and magnesium and a good source of fibre and protein all of which are beneficial in promoting the overall health of pregnant women.Moringa oleifera is a plant that may offer health benefits, including reducing your risk of certain health conditions like high blood pressure.Moringa leaves are rich in many important nutrients, including protein, vitamin B6, vitamin C, riboflavin and iron.Moringa oleifera is rich in various antioxidants, including quercetin and chlorogenic acid."],
        "Elephantyam": ["/static/images/elephantyam.jpg","Elephant foot yam or Kanda or Suran or Zimmikanda is a tropical tuber cash crop that is cultivated extensively in India, Africa, South Asia, Southeast Asia and the tropical Pacific islands.Kanda is a carb and protein rich vegetable, loaded with zinc, phosphorous, potassium, Vitamin B6, Vitamin A and calcium. It also constitutes phenols, alkaloids, flavonoids which play a major role in proper body functions.  It also contains negligible amounts of fat and is water rich.Regular consumption of Kanda vegetable brings down to the levels of LDL or bad cholesterol, thanks to the presence of Omega-3 fatty acids. This tuber also increases the levels of good cholesterol and since it is very low on fat, it can be consumed regularly by heart patients."],
        "Frenchbeans": ["/static/images/frenchbeans.jpg","French beans, being low on calories and high in essential nutrients, can be taken regularly by those people who are strictly following a diet regime to lose weight, especially in the case of those with diabetes. These nutritious veggies also provide dietary fibres that can be processed easily in the stomach, keeping one feeling full for longer, reducing cravings and assisting in burning fat at a quicker pace.Having negligible cholesterol content, French beans can be safely consumed in the diet regularly for promoting heart health. French beans stimulate the normal elimination of body wastes via the excretory system in the body. It boosts the secretion of fluids within the kidneys, promptly getting rid of accumulated toxins and at the same time, guaranteeing proper hydration of the internal organs in the body."],
        "Greenchilli": ["/static/images/Greenchilli.jpg","Green chillies are a rich source of many nutrients. The bioactive compounds include alkaloids, flavonoids, phenolics, essential oils, tannins, steroids, and capsaicin.Green chillies contain a chemical called capsaicin. Capsaicin is an active ingredient responsible for numerous health benefits. Intake of green chillies could help reduce blood sugar levels, as indicated by a human trial.Green chillies may help reduce the buildup of body fat as per animal and human trials. They may improve fat metabolism. Regular intake of green chillies could help reduce body weight and help improve the metabolism of accumulated fat."],
        "Ivygourd": ["/static/images/ivygourd.jpg","Ivy Gourd (Kundru) is a herb used for food and medicinal purposes.Ivy gourd also contains phytonutrients that give cardiac and anti-cancer benefits, such as saponins, flavonoids, and terpenoids.In Ayurvedic medicines, ivy gourd is used to cure diabetes, Cooked and eaten or added to soups are the stems of this climber plant and the leaves.Ivy gourd has vitamins like B2 that are water-soluble. In preserving your energy levels, this vitamin plays a major role."],
        "Kidneybeans": ["/static/images/kidneybeans.jpg","Kidney beans contain healthy proteins, minerals, and vitamins. Eating them can help with weight management, intestinal wellness, and regulating blood sugar.Kidney beans are among the best sources of plant-based protein. They’re also rich in healthy fibers, which moderate blood sugar levels and promote colon health.Kidney beans are a good source of several vitamins and minerals, such as molybdenum, folate, iron, copper, manganese, potassium, and vitamin K1.Kidney beans contain a variety of bioactive plant compounds. Phytohaemagglutinin is a toxic lectin only found in raw or improperly cooked kidney beans."],
        "Ladiesfinger": ["/static/images/ladiesfinger.jpg","Ladyfinger, popularly known as bhindi in India, is rich in nutrients. It is considered a good source of carbohydrates, proteins, vitamins, enzymes, calcium, potassium and many other nutrients.Ladyfinger contains probiotics (good bacteria) that are stomach bacteria’s friends. Ladyfinger may show positive effects on the microbiome (community of good bacteria) in the intestine, as it helps in vitamin B complex biosynthesis. Ladyfinger may produce the same effects as yoghurt in the small intestine."],
        "Lemon": ["/static/images/lemon.jpg","Lemons are a good source of vitamin C.Lemons are high in heart-healthy vitamin C and several beneficial plant compounds that may lower cholesterol.Lemons contain some iron, but they primarily prevent anemia by improving your absorption of iron from plant foods.Lemons are made up of about 10% carbs, mostly in the form of soluble fiber and simple sugars.Antioxidants may help prevent free radicals from causing cell damage that can lead to cancer. However, exactly how antioxidants can help prevent cancer remains unclear."],
        "Mint": ["/static/images/mint.jpg","Mint is a fragrant and delicious plant that is an ingredient in many foods and beverages. It also has health benefits that may include relieving indigestion, improving brain function, and masking bad breath.Mint is a particularly good source of vitamin A, a fat-soluble vitamin that is critical for eye health and night vision.It is also a potent source of antioxidants, especially when compared to other herbs and spices. Antioxidants help protect your body from oxidative stress, a type of damage to cells caused by free radicals.Mint may also be effective at relieving other digestive problems such as upset stomach and indigestion.Many people believe menthol is an effective nasal decongestant that can get rid of congestion and improve airflow and breathing."],
        "Papaya": ["/static/images/papaya.jpg","Papayas are tropical fruit high in vitamin C and antioxidants. Certain compounds in papayas may have anticancer properties and improve heart health, among other health benefits.Papayas also contain healthy antioxidants known as carotenoids — particularly one type called lycopene.Antioxidants, including the carotenoids found in papayas, can neutralize free radicals. lycopene in papaya can reduce cancer risk.It may also be beneficial for people who are being treated for cancer.The antioxidants in papaya may protect your heart and enhance the protective effects of “good” HDL cholesterol.The papain enzyme in papaya can make protein easier to digest.Papaya is a delicious fruit that is best enjoyed ripe. It can be eaten alone or easily combined with other foods."],
        "Pumpkin": ["/static/images/pumpkin.jpg","Pumpkin has an impressive nutrient profile.Besides being packed with vitamins and minerals, pumpkin is also relatively low in calories, as it’s 94% water.Pumpkin is high in vitamins and minerals while being low in calories. It’s also a great source of beta-carotene, a carotenoid that your body converts into vitamin A.Pumpkins contain antioxidants, such as alpha-carotene, beta-carotene and beta-cryptoxanthin. These can neutralize free radicals, stopping them from damaging your cells.Pumpkin is also high in vitamin C, which has been shown to increase white blood cell production, help immune cells work more effectively and make wounds heal faster."],
        "Radish": ["/static/images/radish.jpg","Radishes have an abundance of vitamins and minerals.Radish contains glucosinolates, which are sulfur-containing compounds found in cruciferous vegetables.Radishes contain a good amount of fibre at 1.9 g per 116 g of vegetables.This vegetable contains anti-diabetic properties that can enhance immune system function, improve glucose uptake and lower blood sugar.Not only is radish extremely hydrating, but the vegetable also contains high levels of vitamin C, a vitamin known to be very beneficial for the skin. Vitamin C improves elasticity of the skin by helping to form collagen, a nutrient that makes up the structure of the skin, bones and other connective tissue.Radishes have a very high water content, 93.5 g per every 100 g! "],
        "Ridgegourd": ["/static/images/ridgegourd.jpg","A vegetable of the gourd family, ridge gourd comprises copious volumes of water. In addition, the flesh of ridge gourd is abundant in cellulose, a natural dietary fibre. Therefore, consuming ridge gourd in dals or fries, or simply drinking a glass of ridge gourd juice sweetened with some honey instantly provides relief from constipation, besides restoring normal bowel movement and digestion.Ridge gourd is bestowed with the ability to purify blood of toxic wastes, alcohol residues and undigested food particles.Ridge gourd is inherently low in calories, sugars, while comprising vast reserves of dietary fibres, which aid in regulating appetite, curbing untimely cravings and controlling weight gain."],
        "Snakegourd": ["/static/images/snakegourd.jpg","Snake gourd, being low on calories and high in essential nutrients, can be taken regularly by those people who are strictly following a diet regimen to lose weight, especially in the case of those with diabetes.Having negligible cholesterol content, snake gourd can be safely consumed in diet regularly for promoting heart health.Snake gourd stimulates the natural elimination of body wastes and kidney stones via the excretory system in the body. It boosts the secretion of fluids within the kidneys, promptly getting rid of accumulated toxins and at the same time, guaranteeing proper hydration of the internal organs in the body."],
        "Sorrel": ["/static/images/sorrel.jpg","Sorrel is a type of leafy green with a sour, lemon-like flavor. It’s used as an herb and a vegetable. There are two main types of sorrel — French and common — which differ slightly in terms of taste and appearance.Sorrel is especially high in vitamin C, a water-soluble vitamin that fights inflammation and plays a key role in immune function.It’s also high in fiber, which can promote regularity, increase feelings of fullness, and help stabilize blood sugar levels.Sorrel is a great source of antioxidants, which are beneficial compounds that protect your cells from damage by neutralizing harmful free radicals.It’s especially popular in soups and stews and often paired with ingredients like potatoes, carrots, chicken, and sour cream."],
        "Spinach": ["/static/images/spinach.jpg","Eating spinach may benefit eye health, reduce oxidative stress, help prevent cancer, and reduce blood pressure levels.Spinach is an extremely nutrient-rich vegetable. It packs high amounts of carotenoids, vitamin C, vitamin K, folic acid, iron, and calcium.Spinach also contains several other vitamins and minerals, including potassium, magnesium, and vitamins B6, B9, and E.Human eyes also contain high quantities of these pigments, which protect your eyes from the damage caused by sunlight.The leafy green is said to contain 250 milligrams of calcium per cup and this would help keep your bones healthy including your teeth.Spinach is said to have high potassium content that is usually recommended for people suffering from high blood pressure. "],
        "Sweetcorn": ["/static/images/sweetcorn.jpg","Sweet Corn has numerous health benefits.  It contains a plethora of nutrients like fibre, antioxidants like carotenoids lutein and zeaxanthin, thiamine (vitamin B1) that helps to keep health concerns at bay. This vegetable can help in digestion, reduces the risks of heart diseases, cognitive decline, cancer and even enhances eye health.The high fibre content is one of the sweet corn benefits. Dietary fibre is beneficial for overall health, including the digestive system. Additionally, it may reduce the risk of heart disease, stroke, type 2 diabetes.One of the sweet corn benefits is that it is rich in fibre. The potential of fibre to decrease blood pressure and cholesterol plays a role in preventing heart disease.Eating corn may promote healthy cognitive (brain) function.   "],
        "Sweetpotato": ["/static/images/sweetpotato.jpg", "Sweet potatoes are nutritious, packing a good amount of vitamin A, vitamin C, and manganese into each serving. They also have anticancer properties and may promote immune function and other health benefits.Sweet potatoes are a great source of fiber, vitamins, and minerals.The fiber and antioxidants in sweet potatoes can be beneficial for gut health.Sweet potatoes are incredibly rich in beta carotene, the antioxidant responsible for the vegetable’s bright orange color.Animal studies have shown that sweet potatoes may improve brain health by reducing inflammation and preventing mental decline. However, it remains unknown whether they have the same effects in humans.Sweet potatoes are an excellent source of beta carotene, which can be converted to vitamin A and help support your immune system and gut health."],
        "Taro": ["/static/images/taro.jpg","Taro root is a vegetable used in a variety of cuisines around the world. It has a mild, nutty taste, starchy texture, and nutrition benefits that make it a healthier alternative to other root vegetables like potatoes.Taro is rich in nutrients that can provide important health benefits. A one-cup serving has a third of your daily recommended intake of manganese, which contributes to good metabolism, bone health, and blood clotting.There are high levels of potassium in taro root, a mineral that helps to control high blood pressure by breaking down excess salt. This reduces stress on your cardiovascular system, helping to prevent development of chronic heart problems. "],
        "Tomato": ["/static/images/tomato.jpg","Tomatoes are the major dietary source of the antioxidant lycopene, which has been linked to many health benefits, including reduced risk of heart disease and cancer.Fresh tomatoes are low in carbs. The carb content consists mainly of simple sugars and insoluble fibers. These fruits are mostly made up of water.Tomatoes are a good source of several vitamins and minerals, such as vitamin C, potassium, vitamin K, and folate.Lycopene is one of the most abundant plant compounds in tomatoes. It’s found in the highest concentrations in tomato products, such as ketchup, juice, paste, and sauce.Tomato-based foods rich in lycopene and other plant compounds may protect against sunburn."],
        "Yellowcucumber": ["/static/images/yellowcucumber.jpg","Yellow cucumber or lemon cucumber scientifically called as the cucumis sativus and known as Dosakai (in Telugu) is a vegetable yellow in color available in parts of India. This versatile cucumber is sweet and flavorful, and doesn't have much of the chemical that makes other cucumbers bitter and hard to digest.The flesh of cucumbers is primarily composed of water but also contains ascorbic acid (vitamin C) and caffeic acid, both of which help soothe skin irritations and reduce swelling. Cucumbers' hard skin is rich in fiber and contains a variety of beneficial minerals including silica, potassium and magnesium.The silica in cucumber is an essential component of healthy connective tissue, which includes muscles, tendons, ligaments, cartilage, and bone."]
    }
    return crop_data[crop_name]