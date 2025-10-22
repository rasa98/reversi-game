import matplotlib.pyplot as plt


def parse_elo_text(data):
    import re

    # Initialize a dictionary to hold the rankings
    elo_dict = {}

    # Split the data into rounds
    rounds = data.strip().split("\n")

    # Iterate through each round
    for round_data in rounds:
        # Extract the Elo ranking from the string
        matches = re.findall(r'Agent: ([\w\s-]+): ([\d.]+)', round_data)
        for agent, elo in matches:
            # Convert elo to a float
            elo_value = int(float(elo))

            # If the agent is not in the dictionary, initialize it with an empty list
            if agent not in elo_dict:
                elo_dict[agent] = []

            # Append the elo value to the agent's list
            elo_dict[agent].append(elo_value)

    # Print the resulting dictionary
    print(elo_dict)

    return elo_dict


data = """
Elo ranking after 5 rounds:
	Agent: azero 200: 1600.9148884955232
	Agent: azero 30: 1453.7461677798406
	Agent: ppo_mlp: 1318.5175588375052
	Agent: ppo_cnn: 1317.4446217693087
	Agent: MinMax minmax GA depth dyn: 1283.13398885329
	Agent: MinMax minmax human depth dynamic: 1275.7357823330522
	Agent: Mcts 500: 1247.808392394841
	Agent: MinMax minmax GA depth 1: 1215.6865576152336
	Agent: Mcts 100: 1158.9458085251842
	Agent: MinMax minmax human - depth 1: 1145.7462501318716
	Agent: ars_mlp: 1017.975087771382
	Agent: Mcts 30: 1010.1612339026532
	Agent: trpo_cnn: 934.2897410344808
	Agent: Random model: 786.1849247510228

Elo ranking after 10 rounds:
	Agent: azero 200: 1672.0517009435655
	Agent: azero 30: 1456.6563603485904
	Agent: ppo_cnn: 1322.867228992523
	Agent: ppo_mlp: 1299.625874931195
	Agent: MinMax minmax GA depth dyn: 1285.732343731438
	Agent: Mcts 500: 1273.7125302162106
	Agent: MinMax minmax human depth dynamic: 1241.5497352448579
	Agent: MinMax minmax GA depth 1: 1210.2978860246783
	Agent: MinMax minmax human - depth 1: 1182.1043362220591
	Agent: Mcts 100: 1160.8046527178888
	Agent: Mcts 30: 1012.7079686397337
	Agent: ars_mlp: 971.4969456025826
	Agent: trpo_cnn: 912.2686717323357
	Agent: Random model: 764.4147688475335

Elo ranking after 15 rounds:
	Agent: azero 200: 1696.5737838197358
	Agent: azero 30: 1513.2180835040213
	Agent: ppo_mlp: 1309.0561836330162
	Agent: ppo_cnn: 1305.4126349478204
	Agent: MinMax minmax human depth dynamic: 1288.8704286184538
	Agent: MinMax minmax GA depth dyn: 1264.4153939956416
	Agent: Mcts 500: 1256.3014702546795
	Agent: MinMax minmax GA depth 1: 1198.9103005312536
	Agent: MinMax minmax human - depth 1: 1188.2196025987878
	Agent: Mcts 100: 1151.3300571635511
	Agent: Mcts 30: 1003.7127240711225
	Agent: ars_mlp: 958.357218457328
	Agent: trpo_cnn: 918.6957371425405
	Agent: Random model: 713.217385457241

Elo ranking after 20 rounds:
	Agent: azero 200: 1716.4115363807864
	Agent: azero 30: 1528.0428093229093
	Agent: MinMax minmax GA depth dyn: 1317.5176401675253
	Agent: MinMax minmax human depth dynamic: 1300.6134987081402
	Agent: ppo_cnn: 1298.0117305137928
	Agent: ppo_mlp: 1296.9920396692519
	Agent: Mcts 500: 1291.6194786699834
	Agent: MinMax minmax GA depth 1: 1157.804925738621
	Agent: MinMax minmax human - depth 1: 1152.1420115804729
	Agent: Mcts 100: 1147.869994932525
	Agent: Mcts 30: 990.6757506162794
	Agent: ars_mlp: 968.047426189022
	Agent: trpo_cnn: 919.2517426629539
	Agent: Random model: 681.2904190429289

Elo ranking after 25 rounds:
	Agent: azero 200: 1724.5876056845857
	Agent: azero 30: 1540.9242851223673
	Agent: MinMax minmax human depth dynamic: 1339.3329331702696
	Agent: ppo_cnn: 1321.7971936363642
	Agent: Mcts 500: 1294.233658842008
	Agent: MinMax minmax GA depth dyn: 1287.2355342632748
	Agent: ppo_mlp: 1270.7860682187356
	Agent: MinMax minmax GA depth 1: 1179.3511879443163
	Agent: Mcts 100: 1157.141431703563
	Agent: MinMax minmax human - depth 1: 1065.913415125896
	Agent: Mcts 30: 1025.5990870302292
	Agent: ars_mlp: 945.5624265462696
	Agent: trpo_cnn: 928.1614337265164
	Agent: Random model: 685.6647431807969

Elo ranking after 30 rounds:
	Agent: azero 200: 1775.4203183769716
	Agent: azero 30: 1511.1102104155934
	Agent: ppo_mlp: 1334.41447959169
	Agent: MinMax minmax human depth dynamic: 1312.8807805161048
	Agent: MinMax minmax GA depth dyn: 1300.4297921699938
	Agent: Mcts 500: 1263.902442436779
	Agent: ppo_cnn: 1258.4351422614939
	Agent: MinMax minmax GA depth 1: 1171.2600718509564
	Agent: Mcts 100: 1171.232028606558
	Agent: MinMax minmax human - depth 1: 1115.197673919738
	Agent: Mcts 30: 1049.9239594054313
	Agent: ars_mlp: 969.3348285722551
	Agent: trpo_cnn: 872.5616917675493
	Agent: Random model: 660.1875843040767

Elo ranking after 35 rounds:
	Agent: azero 200: 1738.3028444664058
	Agent: azero 30: 1503.4129235057637
	Agent: ppo_mlp: 1359.7049158637674
	Agent: MinMax minmax GA depth dyn: 1341.2225930732648
	Agent: MinMax minmax human depth dynamic: 1330.535113629194
	Agent: ppo_cnn: 1314.9499173673255
	Agent: Mcts 500: 1277.106373747351
	Agent: Mcts 100: 1235.9798310661233
	Agent: MinMax minmax GA depth 1: 1155.0376792958316
	Agent: MinMax minmax human - depth 1: 1091.4328535170187
	Agent: Mcts 30: 985.3250945615299
	Agent: ars_mlp: 940.5589116695127
	Agent: trpo_cnn: 867.3538310403047
	Agent: Random model: 625.3681213917977

Elo ranking after 40 rounds:
	Agent: azero 200: 1768.7293991297904
	Agent: azero 30: 1497.6812779649747
	Agent: ppo_mlp: 1356.6253759378803
	Agent: MinMax minmax GA depth dyn: 1336.7754193042067
	Agent: ppo_cnn: 1330.8558375662333
	Agent: MinMax minmax human depth dynamic: 1313.9558786464559
	Agent: Mcts 500: 1260.455261335015
	Agent: MinMax minmax human - depth 1: 1171.5410539982147
	Agent: MinMax minmax GA depth 1: 1161.2323184933737
	Agent: Mcts 100: 1121.1379724309024
	Agent: Mcts 30: 1028.2705553373228
	Agent: ars_mlp: 964.1599191935031
	Agent: trpo_cnn: 851.8056498424078
	Agent: Random model: 603.0650850149101

Elo ranking after 45 rounds:
	Agent: azero 200: 1740.402426050231
	Agent: azero 30: 1542.6873670186535
	Agent: ppo_cnn: 1380.5538618925198
	Agent: ppo_mlp: 1342.9364513432672
	Agent: MinMax minmax GA depth dyn: 1324.8476075609901
	Agent: MinMax minmax human depth dynamic: 1309.5595962078087
	Agent: Mcts 500: 1287.502615659658
	Agent: Mcts 100: 1127.163358164511
	Agent: MinMax minmax GA depth 1: 1096.115028911491
	Agent: MinMax minmax human - depth 1: 1086.1125935119564
	Agent: ars_mlp: 1050.1432126042928
	Agent: Mcts 30: 1013.4531224155965
	Agent: trpo_cnn: 849.2206999515095
	Agent: Random model: 615.5930629027054

Elo ranking after 50 rounds:
	Agent: azero 200: 1749.1570679963882
	Agent: azero 30: 1520.5137285353162
	Agent: ppo_mlp: 1358.1099925895323
	Agent: ppo_cnn: 1337.2832503001443
	Agent: MinMax minmax GA depth dyn: 1321.3948829959836
	Agent: MinMax minmax human depth dynamic: 1303.3804795785343
	Agent: Mcts 500: 1264.3141138204937
	Agent: Mcts 100: 1159.3257748421552
	Agent: MinMax minmax GA depth 1: 1152.255552159404
	Agent: MinMax minmax human - depth 1: 1093.5428237824299
	Agent: Mcts 30: 1013.83375757653
	Agent: ars_mlp: 991.5159673365471
	Agent: trpo_cnn: 893.8476618044176
	Agent: Random model: 607.8159508773139

Elo ranking after 55 rounds:
	Agent: azero 200: 1767.1596118946975
	Agent: azero 30: 1489.106838521974
	Agent: ppo_mlp: 1390.7930489994067
	Agent: MinMax minmax GA depth dyn: 1336.4359962040623
	Agent: Mcts 500: 1305.1660595133012
	Agent: ppo_cnn: 1295.101241318615
	Agent: MinMax minmax human depth dynamic: 1277.4876268880432
	Agent: MinMax minmax GA depth 1: 1156.6353738437858
	Agent: Mcts 100: 1090.416169630189
	Agent: MinMax minmax human - depth 1: 1085.735057719734
	Agent: Mcts 30: 1022.0387299991978
	Agent: ars_mlp: 992.971805275609
	Agent: trpo_cnn: 898.3078791381803
	Agent: Random model: 658.9355652483925

Elo ranking after 60 rounds:
	Agent: azero 200: 1773.0222410194158
	Agent: azero 30: 1510.5610390124857
	Agent: ppo_mlp: 1342.083701744252
	Agent: MinMax minmax human depth dynamic: 1329.060724263053
	Agent: ppo_cnn: 1328.532492840026
	Agent: Mcts 500: 1292.874566568513
	Agent: MinMax minmax GA depth dyn: 1284.7496164154416
	Agent: Mcts 100: 1154.639231630876
	Agent: MinMax minmax GA depth 1: 1129.5675258905676
	Agent: MinMax minmax human - depth 1: 1120.5746055066916
	Agent: Mcts 30: 982.1665905768624
	Agent: ars_mlp: 970.0487249512253
	Agent: trpo_cnn: 928.5422638166583
	Agent: Random model: 619.8676799591199

Elo ranking after 65 rounds:
	Agent: azero 200: 1747.9557190983692
	Agent: azero 30: 1513.7450326659455
	Agent: MinMax minmax GA depth dyn: 1300.5203368384643
	Agent: MinMax minmax human depth dynamic: 1298.8566987722456
	Agent: ppo_mlp: 1295.1327505395764
	Agent: Mcts 500: 1276.9735191124814
	Agent: ppo_cnn: 1276.9643428340187
	Agent: Mcts 100: 1153.4119858585098
	Agent: MinMax minmax human - depth 1: 1146.0216958356225
	Agent: MinMax minmax GA depth 1: 1142.5158115646072
	Agent: Mcts 30: 1020.1686343733454
	Agent: ars_mlp: 978.0404217830754
	Agent: trpo_cnn: 975.4047650320732
	Agent: Random model: 640.5792898868555

Elo ranking after 70 rounds:
	Agent: azero 200: 1738.2722085210194
	Agent: azero 30: 1503.902703348786
	Agent: ppo_cnn: 1356.7383135957093
	Agent: ppo_mlp: 1305.2013785454296
	Agent: Mcts 500: 1302.8803777535336
	Agent: MinMax minmax human depth dynamic: 1301.5756609472762
	Agent: MinMax minmax GA depth dyn: 1280.0314997680027
	Agent: MinMax minmax GA depth 1: 1154.9628286120485
	Agent: Mcts 100: 1153.760169660022
	Agent: MinMax minmax human - depth 1: 1134.0109168569725
	Agent: Mcts 30: 1032.0858401819162
	Agent: ars_mlp: 949.9374918559994
	Agent: trpo_cnn: 914.2019822973899
	Agent: Random model: 638.7296322510831

Elo ranking after 75 rounds:
	Agent: azero 200: 1781.3032741043205
	Agent: azero 30: 1540.7100789867054
	Agent: MinMax minmax human depth dynamic: 1329.0025654838228
	Agent: ppo_mlp: 1309.1187117375227
	Agent: ppo_cnn: 1303.564091793826
	Agent: MinMax minmax GA depth dyn: 1300.2953484978123
	Agent: Mcts 500: 1297.373746220085
	Agent: MinMax minmax GA depth 1: 1151.5835473863922
	Agent: MinMax minmax human - depth 1: 1135.261680784144
	Agent: Mcts 100: 1098.191715650666
	Agent: ars_mlp: 1022.5409700178742
	Agent: Mcts 30: 974.8209648851721
	Agent: trpo_cnn: 892.2799094067013
	Agent: Random model: 630.2443992401434

Elo ranking after 80 rounds:
	Agent: azero 200: 1792.939144656819
	Agent: azero 30: 1597.0064143636034
	Agent: ppo_cnn: 1334.2470437572092
	Agent: ppo_mlp: 1315.6719923620794
	Agent: MinMax minmax GA depth dyn: 1290.4667692817516
	Agent: Mcts 500: 1285.1636403020857
	Agent: MinMax minmax human depth dynamic: 1284.4734008535404
	Agent: MinMax minmax GA depth 1: 1145.6407854399872
	Agent: MinMax minmax human - depth 1: 1138.4980645679284
	Agent: Mcts 100: 1134.309668844222
	Agent: Mcts 30: 984.0617920728721
	Agent: ars_mlp: 941.7655314594158
	Agent: trpo_cnn: 879.1968017633469
	Agent: Random model: 642.8499544703294

Elo ranking after 85 rounds:
	Agent: azero 200: 1821.620895602215
	Agent: azero 30: 1581.2442055261597
	Agent: ppo_cnn: 1330.842267904801
	Agent: ppo_mlp: 1325.6294289821333
	Agent: Mcts 500: 1312.9182850552904
	Agent: MinMax minmax GA depth dyn: 1284.5577570916323
	Agent: MinMax minmax human depth dynamic: 1276.7753043986988
	Agent: Mcts 100: 1155.0488785289974
	Agent: MinMax minmax GA depth 1: 1134.3070277810882
	Agent: MinMax minmax human - depth 1: 1066.6866959803194
	Agent: Mcts 30: 996.9228109183035
	Agent: ars_mlp: 930.9464741111644
	Agent: trpo_cnn: 913.1612706582918
	Agent: Random model: 635.6297016560948

Elo ranking after 90 rounds:
	Agent: azero 200: 1822.6371511932816
	Agent: azero 30: 1542.1564808382095
	Agent: ppo_cnn: 1359.8331913854456
	Agent: ppo_mlp: 1327.372884622355
	Agent: MinMax minmax GA depth dyn: 1287.0601374066225
	Agent: Mcts 500: 1284.2524102393772
	Agent: MinMax minmax human depth dynamic: 1280.6673584733683
	Agent: Mcts 100: 1135.528387165291
	Agent: MinMax minmax human - depth 1: 1132.6022168681266
	Agent: MinMax minmax GA depth 1: 1112.3907210746465
	Agent: Mcts 30: 1048.0123134447986
	Agent: ars_mlp: 943.2503447846512
	Agent: trpo_cnn: 875.7596082627858
	Agent: Random model: 614.7677984362316

Elo ranking after 95 rounds:
	Agent: azero 200: 1807.5878070933472
	Agent: azero 30: 1522.2969688774183
	Agent: ppo_mlp: 1374.6400825821936
	Agent: MinMax minmax GA depth dyn: 1313.7821332382475
	Agent: ppo_cnn: 1299.6825164980687
	Agent: Mcts 500: 1273.725329096277
	Agent: MinMax minmax human depth dynamic: 1267.5035857595851
	Agent: MinMax minmax GA depth 1: 1193.57854939034
	Agent: MinMax minmax human - depth 1: 1166.5283509423589
	Agent: Mcts 100: 1112.2222621171143
	Agent: Mcts 30: 994.433863909055
	Agent: ars_mlp: 957.7801172608773
	Agent: trpo_cnn: 864.373882922699
	Agent: Random model: 618.155554507611

Elo ranking after 100 rounds:
	Agent: azero 200: 1781.9338600725694
	Agent: azero 30: 1550.8624238058323
	Agent: ppo_cnn: 1335.9891429997433
	Agent: Mcts 500: 1328.3900093288796
	Agent: ppo_mlp: 1320.4062882483965
	Agent: MinMax minmax human depth dynamic: 1311.2850424662402
	Agent: MinMax minmax GA depth dyn: 1252.2537007956064
	Agent: MinMax minmax human - depth 1: 1191.2549411903144
	Agent: MinMax minmax GA depth 1: 1152.7328989135058
	Agent: Mcts 100: 1139.5827555564504
	Agent: Mcts 30: 996.5728334735885
	Agent: ars_mlp: 916.6056796228017
	Agent: trpo_cnn: 873.2267673103408
	Agent: Random model: 615.1946604109222
"""

agent_dict = {
    "azero 200": "azero200",
    "azero 30": "azero30",
    "ppo_cnn": "ppo_cnn",
    "Mcts 500": "Mcts500",
    "ppo_mlp": "ppo_mlp",
    "MinMax minmax human depth dynamic": "MM_human_dyn",
    "MinMax minmax GA depth dyn": "MM_GA_dyn",
    "MinMax minmax human - depth 1": "MM_human_d1",
    "MinMax minmax GA depth 1": "MM_GA_d1",
    "Mcts 100": "Mcts100",
    "Mcts 30": "Mcts30",
    "ars_mlp": "ars_mlp",
    "trpo_cnn": "trpo_cnn",
    "Random model": "Random"
}

# Data from the provided input
rounds = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
elo_ratings = parse_elo_text(data)

colors = [
    'b', 'g', 'r', 'c', 'm', 'y', 'k', 'purple', 'orange', 'pink',
    'gray', 'brown', 'olive', 'navy', 'teal'
]

# Define an extended list of markers for better distinction
markers = [
    'o', 'v', 's', 'P', 'D', 'X', '*', 'd', '^', '<', '>',
    'H', '8', 'p'
]

plt.figure(figsize=(12, 8))

for idx, (agent, ratings) in enumerate(elo_ratings.items()):
    agent = agent_dict[agent]
    plt.plot(rounds, ratings, color=colors[idx % len(colors)], marker=markers[idx % len(markers)], label=agent)

plt.title('Elo Rating Progression Over Rounds')
plt.xlabel('Rounds')
plt.ylabel('Elo Rating')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.show()
