import os, pickle, re
import numpy as np
import pandas as pd
import CMUTweetTagger
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet
from collections import Counter
import itertools
from training_params import get_general_params
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix, hstack
from training_utils import *

abuse_indicating_terms = ['alcohol', 'coffee', 'white', 'red', 'wine', 'vodka',
    'shots', 'patron', 'booze', 'margarita', 'mimosa', 'xanax', 'painkiller', 'caffeine',
    'alcohol', 'happy pills', 'adderall', 'concerta', 'cocaine', 'rum', 'enough', 'pop', 
    'popping', 'not enough', 'another', 'test', 'final', 'study', 'studying', 'problems',
    'college', 'class', 'breakfast', 'rely', 'sleep', 'sleeping', 'work', 'family problems', 
    'stressful', 'stress', 'skinny', 'snort', 'crush', 'inject', 'inhale', 'finals', 'studi',
    'midterm', 'exam', 'homework', 'paper', 'essay', 'project', 'school', 'cram', 'quiz', 
    'assignment', 'all-night', 'allnight', 'booz', 'beer', 'drink', 'bud', 'red bull', 
    'monster', 'no dose', 'no doze', '5 hour energy', 'five hour energy', 'rockstar', 
    'coke', 'crack', 'rock', 'freebase', 'marijuana', 'MJ', 'pot', 'weed', 'grass', 
    'reefer', 'Mary Jane', 'tranquilizer', 'valium', 'beanies', 'ativan', 'benzo', 'crystal',
    'meth', 'methamphetamine', 'amphetamine']

drug_slang_lexicon = ['a-bomb', 'a-boot', 'abandominiums', "abe's cabe", 'abolic', 'acapulco gold', 'acapulco red', 'acid cube', 'acid freak', 'acid head', 'aeon flux', 'afgani indica', 'african', 'african black', 'african bush', 'african woodbine', 'agonies', 'ah-pen-yen', 'aimies', 'air blast', 'airhead', 'airplane', 'al capone', 'all lit up', 'all star', 'all-american drug', 'alpha-et', 'amidone', 'amoeba', 'amp head', 'amp joint', 'amped-out', 'amphets', 'amping', 'anadrol', 'anatrofin', 'anavar', 'angel dust', 'angel hair', 'angel mist', 'angel poke', 'angel powder', 'angola', 'animal', 'animal trank', 'animal tranq', 'animal tranquilizer', 'antifreeze', 'apache', 'apple jacks', 'arnolds', 'aroma of men', 'around the turn', 'artillery', 'aspirin', 'assassin of youth', 'astro turf', 'atom bomb', 'atshitshi', 'aunt hazel', 'aunt mary', 'aunt nora', 'aunti emma', 'aurora borealis', 'author', 'b-bombs', 'baby bhang', 'baby habit', 'baby t', 'babysit', 'babysitter', 'back breakers', 'back dex', 'back door', 'back jack', 'back to back', 'backtrack', 'backup', 'backwards', 'bad bundle', 'bad go', 'bad seed', 'badrock', 'bag bride', 'bag man', 'bagging', 'balling', 'balloon', 'ballot', 'bambalacha', 'bambita', 'bammies', 'banana split', 'banano', 'banging', 'bank bandit pills', 'barbies', 'barnyard hay', 'barrels', 'bart simpson', 'base crazies', 'base head', 'baseball', 'based out', 'basing', 'bathtub crank', 'bathtub speed', 'batman', 'batman', 'batmans', 'batted out', 'battery acid', 'bazooka', 'bazulco', 'bc bud', 'bdmpea', 'beam me up scottie', 'beam me up scotty', 'beamer', 'beamers', 'beannies', 'beat artist', 'beat vials', 'beautiful boulders', 'bed bugs', 'bedbugs', 'beedies', 'beemers', 'behind the scale', 'beiging', 'belladonna', 'belted', 'belushi', 'belyando spruce', 'bender', 'bennie', 'bennies', 'benzedrine', 'benzidrine', 'bermuda triangles', 'bernice', 'bernie', "bernie's flakes", "bernie's gold dust", 'bickie', 'big bag', 'big bloke', 'big doodig', 'big flake', 'big harry', 'big man', 'big rush', 'bikers coffee', 'bill blass', 'billie hoke', 'bin laden', 'bindle', 'bingers', 'biphetamine', 'bipping', 'birdhead', 'birdie powder', 'biscuit', "bite one's lips", 'black acid', 'black and white', 'black bart', 'black beauties', 'black beauty', 'black birds', 'black bombers', 'black cadillacs', 'black dust', 'black eagle', 'black ganga', 'black gold', 'black grandma', 'black gungi', 'black gunion', 'black hash', 'black hole', 'black mollies', 'black mote', 'black pearl', 'black pill', 'black rock', 'black russian', 'black star', 'black stuff', 'black sunshine', 'black tabs', 'black tar', 'black whack', 'blacks', 'blanket', 'blanks', 'blast a joint', 'blast a roach', 'blast a stick', 'blasted', 'blaxing', 'blazing', 'bling bling', 'blizzard', 'block busters', 'blonde', 'blotter', 'blotter acid', 'blotter cube', 'blow a stick', 'blow blue', 'blow coke', "blow one's roof", 'blow smoke', 'blow the vein', 'blow up', 'blow your mind', 'blowcaine', 'blowing smoke', 'blowout', 'blue acid', 'blue angels', 'blue bag', 'blue barrels', 'blue birds', 'blue boy', 'blue bullets', 'blue caps', 'blue chairs', 'blue cheers', 'blue clouds', 'blue de hue', 'blue devil', 'blue devils', 'blue dolls', 'blue heaven', 'blue heavens', 'blue ice', 'blue kisses', 'blue lips', 'blue madman', 'blue magic', 'blue meth', 'blue microdot', 'blue mist', 'blue mollies', 'blue moons', 'blue nile', 'blue nitro vitality', 'blue sage', 'blue sky blond', 'blue star', 'blue tips', 'blue vials', 'bob hope', 'bobby brown', 'bobo bush', 'body-packer', 'body-stuffer', 'bogart a joint', 'bolasterone', 'bolivian marching powder', 'bomb squad', 'bomber', 'bombido', 'bombs away', 'bonecrusher', 'boo boo bama', 'boomers', 'boomers', 'boost and shoot', 'booster', 'boot the gong', 'booted', 'bopper', 'boppers', 'botray', 'bottles', 'boubou', 'boulder', 'boulya', 'bouncing powder', 'box labs', 'boy-girl', 'brain damage', 'brain pills', 'brain ticklers', 'break night', 'breakdown', 'brewery', 'brick gum', 'bridge or bring up', 'britton', 'broccoli', 'broker', 'brown bombers', 'brown crystal', 'brown dots', 'brown rhine', 'brown sugar', 'brown tape', 'brownies', 'browns', 'bubble gum', 'bubbler', 'buddha', 'buffer', 'bugged', 'bull dog', 'bulladine', 'bullet', 'bullet bolt', 'bullia capital', 'bullion', 'bullyon', 'bumblebees', 'bummer trip', 'bump up', 'bumper', 'bumping up', 'bundle', 'burese', 'burn one', 'burn the main line', 'burn transaction', 'burned', 'burned out', 'burnese', 'burnie', 'burnout', "businessman's lsd", "businessman's special", "businessman's trip", 'busted', 'busters', 'busy bee', 'butler', 'butt naked', 'butter', 'butter flower', 'buttons', 'buzz bomb', 'c joint', 'c-dust', 'c-game', 'cabbage head', 'cactus', 'cactus buttons', 'cactus head', 'cadillac', 'cadillac express', 'cafeteria use', 'cafeteria-style use', 'california cornflakes', 'california sunshine', 'cam trip', 'canade', 'canadian black', 'canamo', 'canappa', 'cancelled stick', 'candy blunt', 'candy c', 'candy flipping on a string', 'candy raver', 'candy sticks', 'candy sugar', 'candy-flipping', 'candyman', 'cannabinol', 'cannabis tea', 'cap up', 'capital h', 'captain cody', 'carburetor', 'care bears', 'carmabis', 'carnie', 'carpet patrol', 'carrie', 'carrie nation', 'cartwheels', 'casper', 'casper the ghost', 'cat in the hats', 'cat killer', 'cat valium', 'catnip', 'caviar', 'cavite all star', 'chalked up', 'chalking', 'champagne', 'channel', 'channel swimmer', 'charas', 'charge', 'charged up', 'charity', 'charley', 'charlie', 'charlie brown', 'chaser', 'chasing the dragon', 'chasing the tiger', 'cheap basing', 'cheeba', 'cheese', 'cheese', 'cheese', 'chemical', 'cherry meth', 'chewies', 'chiba chiba', 'chicago black', 'chicago green', 'chicken feed', 'chicken powder', 'chicken scratch', 'chiefing', 'chieva', 'chillum', 'china cat', 'china girl', 'china town', 'china white', 'chinese molasses', 'chinese red', 'chinese tobacco', 'chipper', 'chipping', 'chippy', 'choco-fan', 'chocolate', 'chocolate chip cookies', 'chocolate chips', 'chocolate ecstasy', 'chocolate rock', 'chocolate thai', 'cholly', 'chorals', 'chowder', 'christina', 'christmas bud', 'christmas rolls', 'christmas tree', 'christmas tree meth', 'chrome', 'chronic', 'chrystal methadrine', 'chucks', 'chunky', 'churus', 'cigamos', 'cigarette paper', 'cigarrode cristal', 'cinnamon', 'circles', 'citrol', 'clam bake', 'clarity', 'clear up', 'clicker', 'clickums', 'cliffhanger', 'climax', 'clocker', 'clocking paper', 'closet baser', 'cloud nine', 'cluckers', 'co-pilot', 'coasting', 'coasts to coasts', 'cocaine blues', 'cochornis', 'cocktail', 'coco rocks', 'coco snow', 'cocoa puff', 'coconut', 'coffee', 'coke bar', 'cokehead', 'cold turkey', 'colombian', 'colorado cocktail', 'columbo', 'columbus black', 'combol', 'come home', 'come up', 'comeback', 'comic book', 'conductor', 'connect', 'connie', 'contact lens', 'cook down', 'cooker', 'cookies', 'cooking up', 'cooler', 'coolie', 'copping zones', 'coriander seeds', 'cork the air', 'corrine', 'corrinne', 'cotics', 'coties', 'cotton', 'cotton brothers', 'cotton fever', 'courage pills', 'course note', 'cousin tina', "cozmo's", 'crack attack', 'crack back', 'crack bash', 'crack cooler', 'crack gallery', 'crack house', 'crack kits', 'crack spot', 'crack-in-the-box', 'cracker jack', 'cracker jacks', 'crackers', 'crangbustin', 'cranking up', 'crankster', 'crazy coke', 'crazy eddie', 'crazy weed', 'credit card', 'cresant roll', 'crimmie', 'cringe', 'cripple', 'crisscross', 'crisscrossing', 'cristy', 'cronic', 'cross tops', 'crossles', 'crossroads', 'crown crap', 'crumbs', 'crush and rush', 'crying weed', 'cryppie', 'crypto', 'cryptonie', 'crystal', 'crystal glass', 'crystal joint', 'crystal meth', 'crystal methadrine', 'crystal t', 'crystal tea', 'culican', 'cupcakes', 'cushion', 'cut-deck', 'cycline', 'cyclones', 'dabble', 'dance fever', 'dawamesk', 'dead on arrival', 'dead president', 'dead road', 'deca-duabolin', 'decadence', 'deisel', 'delatestryl', 'demolish', 'dep-testosterone', 'desocsins', 'desogtion', 'detroit pink', 'devil drug', "devil's bush", "devil's dandruff", "devil's dick", "devil's dust", 'devilsmoke', 'dexedrine', 'dexies', 'diambista', 'diamond folds', 'diamonds', 'dianabol', 'diesel', 'diet pills', 'dihydrolone', 'dime bag', 'dime special', "dime's worth", 'dimebag', 'dinkie dow', 'dinosaurs', 'dipped joints', 'dipper', 'dipping out', 'dirt grass', 'dirties', 'dirty basing', 'dirty dirt', 'dirty joints', 'disco biscuit', 'disco biscuits', 'disco pellets', 'discorama', 'disease', 'ditch weed', "diviner's sage", 'djamba', 'do a joint', 'do a line', 'do it jack', 'doctor', 'doctor shopping', 'dog food', 'dolla boy', 'dollar', 'domestic', 'dominican knot', 'dominoes', 'don jem', 'don juan', 'donkey', 'donnie brasco', 'doobee', 'doobie', 'dooley', 'doosey', 'dope fiend', 'dope smoke', 'dopium', 'doradilla', "dors and 4's", 'dosure', 'double breasted dealing', 'double bubble', 'double cross', 'double dome', 'double rock', 'double trouble', 'double up', 'double ups', 'double yoke', "dover's deck", "dover's powder", 'downer', 'downie', 'draf weed', 'drag weed', 'dragon rock', 'draw up', 'dream gun', 'dream stick', 'dreamer', 'dreams', 'drivers', 'dropper', 'dropping', 'drought', 'drowsy high', 'dry high', 'dry up', 'dummy dust', 'durabolin', 'durong', 'dust blunt', 'dust joint', 'dust of angels', 'dusted parsley', 'dusting', 'dymethzine', 'dynamite', 'dyno-pure', 'e-bombs', 'e-puddle', 'e-tard', 'easing powder', 'eastside player', 'easy lay', 'easy score', 'eating', 'ecstasy', 'egyptians', 'eight ball', 'eight-ball', 'eightball', 'eighth', 'elbows', 'electric kool aid', 'electric kool-aid', 'elephant', 'elephant flipping', 'elephant trank', 'elephant tranquilizer', 'elephants', 'embalming fluid', 'emergency gun', 'energizer', 'enoltestovis', 'ephedrone', 'equipose', 'essence', 'estuffa', 'everclear', 'exiticity', 'explorers club', 'eye opener', 'eye openers', 'factory', 'fake stp', 'fallbrook redhair', 'famous dimes', 'fantasia', 'fantasy', 'fast white lady', 'fastin', 'fat bags', 'fattie', 'feed bag', 'feeling', 'feenin', 'felix the cat', 'ferry dust', 'fi-do-nie', 'fields', 'fifteen cents', 'fifty-one', 'fine stuff', 'finger', 'finger lid', 'fingers', 'fire it up', 'fireflower', 'firewater', 'firewood', 'first line', 'fish scales', 'five c note', 'five cent bag', 'five dollar bag', 'five-way', 'fizzies', 'flakes', 'flame cooking', 'flamethrowers', 'flat blues', 'flat chunks', 'flatliners', 'flea powder', 'fleece', 'flipping', 'florida snow', 'flower', 'flower flipping', 'flower tops', 'flowers', 'fly mexican airlines', 'flying', 'following that cloud', 'foo foo', 'foo foo stuff', 'foo-foo dust', 'foolish powder', 'footballs', 'forget me drug', 'forget pill', 'forget-me pill', 'forwards', 'four leaf clover', 'freebase', 'freebasing', 'freeze', 'french blue', 'french fries', 'friend', 'frisco special', 'frisco speedball', 'friskie powder', 'frontloading', 'fry daddy', 'fry sticks', 'g-riffic', 'g-rock', 'g-shot', 'gaffel', 'gaffle', 'gaffus', 'gagers', 'gaggers', 'gaggler', 'galloping horse', 'gallup', 'gamma oh', 'gangster', 'gangster pills', 'ganoobies', 'garbage', 'garbage heads', 'garbage rock', 'gasper', 'gasper stick', 'gauge butt', 'geek-joints', 'geeker', 'geeter', 'geezer', 'geezin a bit of dee gee', 'george', 'george smack', 'georgia home boy', 'get a gage up', 'get a gift', 'get down', 'get high', 'get lifted', 'get off', 'get off houses', 'get the wind', 'get through', 'getting glassed', 'getting roached', 'getting snotty', 'ghostbusting', 'gick monster', 'gift-of-the-sun', 'gift-of-the-sun-god', 'giggle smoke', 'giggle weed', 'gimmick', 'gimmie', 'girlfriend', 'giro house', 'give wings', 'glacines', 'glad stuff', 'glading', 'glass gun', 'go fast', 'go into a sewer', 'go loco', 'go on a sleigh ride', 'go-between', 'go-fast', 'goblet of jam', "god's drug", "god's flesh", "god's medicine", 'gold dust', 'gold star', 'golden', 'golden dragon', 'golden eagle', 'golden girl', 'golden leaf', 'golf ball', 'golf balls', 'gondola', 'good and plenty', 'good butt', 'good giggles', 'good go', 'good h', 'good horse', 'good lick', 'good stuff', 'goodfellas', 'goody-goody', 'goof butt', 'goofball', 'goofers', "goofy's", 'goon dust', 'gopher', 'gorilla biscuits', 'gorilla pills', 'gorilla tab', 'got it going on', 'graduate', 'granulated orange', 'grape parfait', 'grass brownies', 'grasshopper', 'gravel', 'grease', 'great bear', 'great hormones at bedtime', 'great tobacco', 'green buds', 'green double domes', 'green dragons', 'green frog', 'green goddess', 'green goods', 'green leaves', 'green single dome', 'green tea', 'green triangles', 'green wedge', 'greenies', 'greens', 'greeter', 'gremmies', 'grey shields', 'griefo', 'griefs', 'grievous bodily harm', 'griffa', 'griffo', 'grizzy', 'groceries', 'ground control', 'gungeon', 'gungun', 'gutter', 'gutter junkie', 'gym candy', 'h - bomb', 'h caps', 'haircut', 'half a football field', 'half elbows', 'half g', 'half load', 'half moon', 'half piece', 'half track', 'half-a-c', 'hamburger helper', 'hammerheading', 'hand-to-hand', 'hand-to-hand man', 'handlebars', 'hanhich', 'hanyak', 'happy cigarette', 'happy drug', 'happy dust', 'happy pill', 'happy powder', 'happy stick', 'happy sticks', 'happy trails', 'hard ball', 'hard candy', 'hard line', 'hard rock', 'hard stuff', 'hardware', 'have a dust', 'haven dust', 'hawaiian', 'hawaiian black', 'hawaiian homegrown hay', 'hawaiian sunshine', 'hawkers', 'hay butt', 'hayron', 'he-man', 'he-she', 'head drugs', 'head light', 'head shop', 'headies', 'heart-on', 'hearts', 'heaven', 'heaven dust', 'heavenly blue', 'heeled', 'hell dust', 'henpecking', 'henry viii', 'herb and al', 'herbal bliss', 'hero of the underworld', 'herone', 'hessle', 'hiagra in a bottle', 'highball', 'highbeams', 'hikori', 'hikuli', 'hillbilly heroin', 'hinkley', 'hippie crack', 'hippieflip', 'hironpon', 'hiropon', 'hit house', 'hit the hay', 'hit the main line', 'hit the needle', 'hit the pit', 'hitch up the reindeers', 'hitter', 'hitters', 'hitting the slopes', 'hitting up', 'holding', 'holiday meth', 'holy terror', 'homegrown', 'homicide', 'honey blunts', 'honey oil', 'honeymoon', 'hong-yen', 'hoodie', 'hooked', 'hooter', 'hopped up', 'horning', 'horse heads', 'horse tracks', 'horse tranquilizer', 'horsebite', 'hospital heroin', 'hot box', 'hot dope', 'hot heroin', 'hot ice', 'hot rolling', 'hot stick', 'hotcakes', 'hotrailing', 'house fee', 'house piece', 'hubba pigeon', 'hubbas', 'huffer', 'huffing', 'hug drug', 'hugs and kisses', 'hulling', 'hunter', 'hustle', 'hyatari', 'hydrogrows', 'hype stick', 'i am back', 'ice cream habit', 'ice cube', 'idiot pills', 'illies', 'illing', 'illy momo', 'inbetweens', 'inca message', 'indian boy', 'indian hay', 'indian hemp', 'indica', 'indonesian bud', 'instaga', 'instagu', 'instant zen', 'interplanetary mission', 'issues', 'jack-up', 'jackpot', 'jackson', 'jam cecil', 'jamaican gold', 'jamaican red hair', 'jay smoke', 'jee gee', 'jefferson airplane', 'jellies', 'jelly baby', 'jelly bean', 'jelly beans', 'jerry garcias', 'jerry springer', 'jet fuel', 'jim jones', 'jive doo jee', 'jive stick', 'joharito', 'johnson', 'jolly bean', 'jolly green', 'jolly pop', 'jonesing', 'joy flakes', 'joy juice', 'joy plant', 'joy pop', 'joy popping', 'joy powder', 'joy smoke', 'joy stick', 'juggle', 'juggler', 'juice joint', 'jumbos', 'junkie', 'junkie kits', 'k-blast', 'k-hole', 'k-lots', 'kabayo', 'kabuki', 'kaksonjae', 'kalakit', 'kangaroo', 'kansas grass', 'karachi', 'kate bush', 'kawaii electric', 'kentucky blue', 'kester plant', 'kick stick', 'kicker', 'kiddie dope', 'killer', 'killer green bud', 'killer joints', 'killer weed', 'kilter', 'kind bud', 'king bud', 'king ivory', 'king kong pills', "king's habit", 'kissing', 'kit kat', 'kitkat', 'kitty flipping', 'kleenex', 'klingons', 'kokomo', 'kona gold', 'krippy', 'kryptonite', 'krystal', 'krystal joint', 'la rocha', 'lactone', 'lady caine', 'lady snow', 'lakbay diva', 'lamborghini', 'lason sa daga', 'late night', 'laugh and scratch', 'laughing gas', 'laughing grass', 'laughing weed', 'lay back', 'lay-out', 'lazy bitch', 'leaky bolla', 'leaky leak', 'leapers', 'leaping', 'legal speed', 'lemon 714', 'lemon drop', 'lemonade', 'letf handed cigarette', 'lethal weapon', 'letter biscuits', 'lettuce', 'lid poppers', 'lid proppers', 'light stuff', 'lightning', 'lime acid', 'liprimo', 'lipton tea', 'liquid e', 'liquid ecstasy', 'liquid g', 'liquid lady', 'liquid x', 'lit up', 'lithium', 'lithium scabs', 'little bomb', 'little boy', 'little ones', 'little smoke', 'live ones', 'llesca', 'load of laundry', 'loaded', 'locker room', 'locoweed', 'loony toons', 'loose shank', 'lou reed', 'loused', 'love affair', 'love boat', 'love drug', 'love flipping', 'love leaf', 'love pearls', 'love pill', 'love pills', 'love trip', 'love weed', 'loveboat', 'lovelies', 'lovely', "lover's speed", "lovers' special", 'lubage', 'lucy in the sky with diamonds', 'luding out', 'lunch money drug', 'macaroni', 'macaroni and cheese', 'machinery', 'maconha', 'mad dog', 'madman', 'magic dust', 'magic mint', 'magic mushroom', 'magic smoke', 'mainline', 'mainliner', 'make up', 'mama coca', 'manhattan silver', 'manteca', 'marathons', 'marching dust', 'marching powder', 'maria pastora', 'marshmallow reds', 'mary and johnny', 'mary ann', 'mary jane', 'mary jonas', 'mary warner', 'mary weaver', 'maryjane', 'maserati', 'matchbox', 'matsakow', 'maui wauie', 'maui-wowie', 'maxibolin', 'mean green', 'medusa', 'meggie', 'mellow yellow', 'mercedes', 'merchandise', 'mescal', 'messorole', 'meth head', 'meth monster', 'meth speed ball', 'methatriol', 'methedrine', 'methlies quik', 'methnecks', 'methyltestosterone', 'mexican brown', 'mexican crack', 'mexican green', 'mexican horse', 'mexican locoweed', 'mexican mud', 'mexican mushrooms', 'mexican red', 'mexican reds', 'mexican speedballs', 'mexican valium', 'mickey finn', "mickey's", 'microdot', 'midnight oil', 'mighty joe young', 'mighty mezz', 'mighty quinn', 'mighty white', 'mind detergent', 'mini beans', 'minibennie', 'mint leaf', 'mint weed', 'miss emma', 'miss emma', 'missile basing', 'mission', 'mister blue', 'mitsubishi', 'mixed jive', 'modams', 'mohasky', 'mohasty', 'money talks', 'monkey', 'monkey dust', 'monkey tranquilizer', 'monkey-dribble', 'monoamine oxidase', 'monster', 'moon gas', 'moonrock', 'moonstone', 'mooster', 'mooters', 'mootie', 'mootos', 'mor a grifa', 'morning shot', 'morning wake-up', 'morotgara', 'morpho', 'mortal combat', 'mosquitos', 'mother', "mother's little helper", 'motorcycle crack', 'mouth worker', 'movie star drug', 'mow the grass', 'muggie', 'muggle', 'muggles', 'murder 8', 'murder one', 'murotugora', 'mushrooms', 'mustard', 'muzzle', 'nailed', 'nazimeth', 'nebbies', 'nemmies', 'new acid', 'new addition', 'new jack swing', 'new magic', 'new one', 'nexus flipping', 'nice and easy', 'nickel', 'nickel bag', 'nickel deck', 'nickel note', 'nickelonians', 'nimbies', 'nine ball', 'nineteen', 'no worries', 'nontoucher', 'northern lights', 'northern lights', 'nose candy', 'nose drops', 'nose powder', 'nose stuff', 'nugget', 'nuggets', 'number', 'number 3', 'number 4', 'number 8', 'oatmeal', 'ocean cities', 'ocean citys', 'octane', 'old garbage', 'old navy', 'old steve', 'on a mission', 'on a trip', 'on ice', 'on the ball', 'on the bricks', 'on the nod', 'one and one', 'one and ones', 'one bomb', 'one on one house', 'one plus one sales', 'one tissue box', 'one way', 'one-fifty-one', 'one-stop shop', 'oolies', 'optical illusions', 'orange bandits', 'orange barrels', 'orange crystal', 'orange cubes', 'orange haze', 'orange line', 'orange micro', 'orange wedges', 'oranges', 'organic quaalude', 'outerlimits', 'outfit', 'owsley', "owsley's acid", 'oxicotten', "oxy 80's", 'oxycet', 'oxycotton', 'oyster stew', 'p and p', 'p-dogs', 'p-dope', 'p-funk', 'pac man', 'pack a bowl', 'pack of rocks', 'pakaloco', 'pakalolo', 'pakistani black', 'panama cut', 'panama gold', 'panama red', 'panatella', 'pancakes and syrup', 'pangonadalot', 'paper acid', 'paper bag', 'paper blunts', 'paper boy', 'paper chaser', 'papers', 'parabolin', 'parachute', 'parachute down', 'paradise', 'paradise white', 'pariba', 'parlay', 'parsley', 'party and play', 'party pack', 'peace pill', 'peace tablets', 'peace weed', 'peaches', 'peanut', 'peanut butter', 'pearls', 'pearly gates', 'pebbles', 'peddlar', 'pedico', 'pee wee', 'pellets', 'pen yan', 'pep pills', 'pepsi habit', 'perc-a-pop', 'percia', 'percio', 'perfect high', 'perico', 'peruvian', 'peruvian flake', 'peruvian lady', 'peter pan', 'peyote', 'peyote', 'pharming', 'phennies', 'phenos', 'philly blunts', 'pianoing', 'picking', 'pig killer', 'piggybacking', 'pikachu', 'pill houses', 'pill ladies', 'pimp your pipe', 'pin gon', 'pin yen', 'ping-in-wing', 'pingus', 'pink blotters', 'pink elephants', 'pink hearts', 'pink ladies', 'pink panther', 'pink panthers', 'pink robots', 'pink wedges', 'pink witches', 'pixies', 'planks', 'playboy bunnies', 'playboys', 'po coke', 'po-fiend', 'pocket rocket', 'poison', 'pollutants', 'pony packs', "poor man's coke", "poor man's heroin", "poor man's pot", 'poppers', 'poppers', 'potato', 'potato chips', 'potlikker', 'potten bush', 'powder', 'powder diamonds', 'power puller', 'predator', 'premos', 'prescription', 'pretendica', 'pretendo', 'primbolin', 'prime time', 'primo square', 'primo turbo', 'primobolan', 'primos', 'product', 'proviron', 'pseudocaine', 'puff the dragon', 'puffer', 'pulborn', 'pullers', 'pumpers', 'pumping', 'pure love', 'purple', 'purple barrels', 'purple caps', 'purple flats', 'purple gel tabs', 'purple haze', 'purple hearts', 'purple ozoline', 'purple pills', 'purple rain', 'push shorts', 'pusher', 'quarter', 'quarter bag', 'quarter moon', 'quarter piece', 'quartz', "queen ann's lace", 'quicksilver', 'quinolone', 'r-ball', 'racehorse charlie', 'ragweed', 'railroad weed', 'rainbow', 'rainbows', 'rainy day woman', 'rangood', 'raspberry', 'rasta weed', 'rave energy', 'raw fusion', 'raw hide', 'ready rock', 'real tops', 'recompress', 'recycle', 'red and blue', 'red bud', 'red bullets', 'red caps', 'red chicken', 'red cross', 'red devil', 'red devils', 'red dirt', 'red eagle', 'red lips', 'red phosphorus', 'red rock', 'red rock opium', 'red rocks', 'red rum', 'red stuff', 'redneck cocaine', 'reefer', 'reefers', 'reindeer dust', 'renewtrient', 'rest in peace', 'reupped', 'revivarant', 'revivarant-g', 'reynolds', 'rhythm', 'richard', 'riding the wave', 'righteous bush', 'ringer', 'rippers', 'ritual spirit', 'ritz and ts', 'roach clip', 'roach-2', 'roacha', 'roaches', 'roachies', 'road dope', 'roapies', 'roasting', "robin's egg", 'robutal', 'rochas dos', 'rock attack', 'rock climbing', 'rock house', 'rock star', 'rocket caps', 'rocket fuel', 'rockets', 'rockette', 'rocks of hell', 'rocky iii', 'roid rage', 'roller', 'rollers', "rollin'", 'rolling', 'rolls royce', 'rompums', 'roofie', 'roofies', 'rooster', 'rophies', 'ropies', 'roples', 'rose marie', 'rough stuff', 'row-shay', 'roxanne', 'roxies', 'royal blues', 'ruderalis', 'ruffies', 'ruffles', 'runners', 'running', 'rush hour', 'rush snappers', 'russian sickles', 'sacrament', 'sacred mushroom', 'salt and pepper', 'salty water', 'sandoz', 'sandwich', 'sandwich bag', 'sasfras', "satan's secret", 'satch cotton', 'sativa', 'scaffle', 'scarecrow', 'schmeck', 'schmiz', 'schoolboy', 'schoolcraft', 'schwagg', 'scissors', 'scooby snacks', 'scootie', 'scorpion', 'scottie', 'scotty', 'scrabble', 'scramble', 'scrape and snort', 'scratch', 'scruples', 'scuffle', 'second to none', 'seconds', 'serial speedballing', 'sernyl', 'serpico 21', 'server', 'seven-up', 'sevenup', 'sextasy', 'sharps', 'shebanging', 'sheet rocking', 'sheets', 'sherm sticks', 'sherman stick', 'shermans', 'shermhead', 'sherms', 'shnizzlefritz', 'shoot the breeze', 'shooting gallery', 'shoppers', 'shot down', 'shot to the curb', 'shotgun', 'shrile', 'shroom', 'shrooms', 'shrubs', 'sightball', 'silly putty', 'silver bullet', 'simple simon', 'sinsemilla', 'sixty-two', 'sketch', 'sketching', 'skin popping', 'skittles', 'skittling', 'skuffle', 'skunkweed', 'slanging', 'sleep-500', 'sleeper', 'sleeper and red devil', 'sleigh ride', 'slick superspeed', 'smears', 'smoke a bowl', 'smoke canada', 'smoke houses', 'smoke-out', 'smoking', 'smoking gun', 'smooch', 'smurfs', 'snackies', 'snappers', 'sniffer bag', 'snorting', 'snorts', 'snotballs', 'snotty', 'snow bird', 'snow coke', 'snow pallets', 'snow seals', 'snow white', 'snowball', 'snowcones', 'snowman', 'snowmen', 'soap dope', 'society high', 'softballs', 'somali tea', 'somatomax', 'sopers', 'south parks', 'space base', 'space cadet', 'space dust', 'space ship', 'spaceball', 'spackle', 'spark it up', 'sparkle', 'sparkle plenty', 'sparklers', 'special k', 'special la coke', 'speckled birds', 'spectrum', 'speed for lovers', 'speed freak', 'speedball', 'speedballing', 'speedballs-nose-style', 'speedboat', 'speedies', 'spider', 'spider blue', 'spirals', 'spivias', 'splash', 'spliff', 'spliff', 'splitting', 'splivins', 'spoosh', 'spores', 'sporos', 'sporting', 'sprung', 'square mackerel', 'square time bob', 'squares', 'squirrel', 'stackers', 'stacking', 'stacks', 'star dust', 'star-spangled powder', 'stardust', 'stash areas', 'steerer', 'step on', 'sticky icky', 'sticky icky', 'stink weed', 'stoned', 'stones', 'stoney weed', 'stoppers', 'stove top', 'strawberries', 'strawberry', 'strawberry fields', 'strawberry shortcake', 'strung out', 'studio fuel', 'stumbler', 'sugar block', 'sugar boogers', 'sugar cubes', 'sugar lumps', 'sugar weed', 'sunshine', 'super acid', 'super c', 'super grass', 'super ice', 'super joint', 'super kools', 'super pot', 'super weed', 'super x', 'superlab', 'superman', 'supermans', 'surfer', 'sustanon 250', 'swallower', 'swedge', 'sweet dreams', 'sweet jesus', 'sweet lucy', 'sweet stuff', 'sweeties', 'sweets', 'swell up', 'swishers', 'synthetic cocaine', 'synthetic tht', 't-buzz', 'tachas', 'tail lights', 'taking a cruise', 'takkouri', 'tardust', 'taxing', 'tea party', 'teardrops', 'tecatos', 'teenager', 'ten pack', 'tension', 'tester', 'tex-mex', 'texas pot', 'texas shoe shine', 'texas tea', 'thai sticks', 'thanie', 'the beast', 'the bomb', 'the devil', 'the five way', 'the ghost', 'the hawk', 'the nasty', 'the witch', 'therobolin', 'thirst monster', 'thirst monsters', 'thirteen', 'thirty-eight', 'thoroughbred', 'thrust', 'thrusters', 'thunder', 'tic tac', 'tick tick', 'ticket', 'timothy leary', 'tissue', 'toke up', 'toliet water', 'tom and jerries', 'tomater', 'toncho', 'tooles', 'toonies', 'tooter', 'tooties', 'tootsie roll', 'top drool', 'top gun', 'torch cooking', 'torch up', 'tornado', 'torpedo', 'toss up', 'toss-ups', 'totally spent', 'toucher', 'touter', 'tracks', 'tragic magic', 'trails', 'trambo', 'trapped vehicles', 'trauma', 'travel agent', 'triple a', 'triple crowns', 'triple folds', 'triple rolexes', 'triple stacks', "trippin'", 'trophobolene', 'truck drivers', 'trupence bag', 'ts and rits', 'ts and rs', 'turkey', 'turnabout', 'turned on', 'tustin', 'twakers', 'tweak mission', 'tweaker', 'tweaking', 'tweaks', 'tweeker', 'tweety birds', 'twenties', 'twenty', 'twenty rock', 'twenty-five', 'twin towers', 'twinkie', 'twisters', 'twists', 'twistum', 'two for nine', 'tyler berry', 'ultimate', 'ultimate xphoria', 'uncle milty', 'unotque', 'up against the stem', 'uppers', 'uppies', 'ups and downs', 'uptown', 'utopiates', 'vidrio', 'vikings', "viper's weed", 'vita-g', 'vitamin a', 'vitamin k', 'vitamin r', 'vodka acid', 'wacky weed', 'wafers', 'waffle dust', 'wake and bake', 'wake ups', 'wasted', 'water-water', 'watercolors', 'wedding bells', 'weed tea', 'weight trainers', 'weightless', 'west coast', 'west coast turnarounds', 'wet sticks', 'whackatabacky', 'wheels', 'when-shee', 'whiffledust', 'whippets', 'white ball', 'white boy', 'white cloud', 'white cross', 'white diamonds', 'white dove', 'white dragon', 'white dust', 'white ghost', 'white girl', 'white girl', 'white horizon', 'white horse', 'white junk', 'white lady', 'white lightning', 'white mosquito', 'white nurse', "white owsley's", 'white powder', 'white russian', 'white stuff', 'white sugar', 'white tornado', 'white-haired lady', 'whiteout', 'whites', 'whiz bang', 'wicked', 'wicky stick', 'wigging', 'wigits', 'wild cat', 'window glass', 'window pane', 'winstrol', 'winstrol v', 'witch hazel', 'wobble weed', 'wolfies', 'wollie', 'wonder star', 'woo blunts', 'woola blunt', 'woolah', 'woolas', 'woolie', 'woolie blunt', 'woolies', 'wooly blunts', 'wooties', 'working', 'working bags', 'working fifty', 'working half', "working man's cocaine", 'wounded', 'wrecking crew', 'x-pills', 'yeah-o', 'yellow', 'yellow bam', 'yellow bullets', 'yellow dimples', 'yellow fever', 'yellow jackets', 'yellow powder', 'yellow submarine', 'yellow sunshine', 'yen pop', 'yen shee suey', 'yen sleep', 'yerba mala', 'yerba mala', 'yerhia', 'yimyom', 'ying yang', 'zacatecas purple', 'zannie', 'zig zag man', 'zombie', 'zombie weed', 'zonked', 'zoomer', 'zoquete']

# read word cluster from file
def read_word_clusters(plusone = False):
    print('reading word clusters')
    with open('./data/word_clusters.txt', 'r', encoding='utf-8') as inf:
        word_clusters = {}
        lines = inf.readlines()
        for line in lines:
            try:
                content = line.strip().rsplit(' ', 1)
                word = content[0].replace('\/', ' ')
                clu = int(content[1])
            except:
                print(line)
                print(content)
            if plusone:
                word_clusters[word] = clu + 1
            else:
                word_clusters[word] = clu
    return word_clusters

# clean html symbols, urls, and usernames
# then tokenized/taggered by CMU tweet tagger
# and stemmed by porter stemmer
# return token lists and tagging results
def clean_tokenize_stem_tweets_baseline(raw_tweets, token_file, tagging_file, force_new=False):
    if os.path.isfile(token_file) and os.path.isfile(tagging_file) and not force_new:
        print('reading tokens and tagging results')
        with open(token_file, 'rb') as inf:
            token_lists = pickle.load(inf)
        with open(tagging_file, 'rb') as inf:
            tagging_results = pickle.load(inf)
        if len(token_lists) == raw_tweets.shape[0] and len(tagging_results) == raw_tweets.shape[0]:
            print('tokens and tagging results are correct')
            return token_lists, tagging_results
    print('preform new tokenization and tagging')
    # or create new one
    stemmer = PorterStemmer()
    url_compiled = re.compile(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'\".,<>?\xab\xbb\u201c\u201d\u2018\u2019]))')
    mention_compiled = re.compile(r'@\w*')
    html_compiled = re.compile(r"&#?\w+;")
    tweets_cleaned = [html_compiled.sub(' ', tweet) for tweet in raw_tweets]
    tweets_cleaned = [url_compiled.sub(' ', tweet) for tweet in tweets_cleaned]
    tweets_cleaned = [mention_compiled.sub(' ', tweet) for tweet in tweets_cleaned]
    tagging_results = CMUTweetTagger.runtagger_parse(tweets_cleaned)
    token_lists = [[stemmer.stem(word_result[0]) for word_result in tweet_result] for tweet_result in tagging_results]
    # save to file
    with open(token_file, 'wb') as outf:
        pickle.dump(token_lists, outf)
    with open(tagging_file, 'wb') as outf:
        pickle.dump(tagging_results, outf)
    return token_lists, tagging_results

# synonym expansion for ml
def synonym_expansion(tagging_results, vocab):
    # generate a set of synonym to be appended behind each tweet
    # the synonym will be vectorized into a separate vector without using ngram
    # target word pos
    print('doing synonym expansion')
    target_tags = set(['N', '^', 'A', 'V'])
    expansion_set = set()
    expansion_lists = []
    expansion_dict = {}
    for tagsets in tagging_results:
        flag= True
        try:
            tokens = [t[0] for t in tagsets]
            tags = [t[1] for t in tagsets]
        except:
            flag = False
            print(tagsets)
        expan = []
        if flag:
            for i in range(len(tokens)):
                if tags[i] in target_tags:
                    if tokens[i] in expansion_dict:
                        word_expan = expansion_dict[tokens[i]]
                    else:
                        if tags[i] == 'N' or tags[i] == '^':
                            syn = wordnet.synsets(tokens[i], pos=wordnet.NOUN)
                        elif tags[i] == 'V':
                            syn = wordnet.synsets(tokens[i], pos=wordnet.VERB)
                        elif tags[i] == 'A':
                            syn = wordnet.synsets(tokens[i], pos=wordnet.ADJ)
                        if len(syn) > 0:
                            exp_words = [s.name().rsplit('.', 2)[0] for s in syn]
                            exp_word_count = [(w, vocab[w]) if w in vocab else (w, 1e32) for w in exp_words]
                            if len(exp_word_count) > 3:
                                exp_word_count = sorted(exp_word_count, key=lambda x:x[1])
                                word_expan = [exp_word_count[j][0] for j in range(3)]
                            else:
                                word_expan = [w_c[0] for w_c in exp_word_count]
                    expan += word_expan
                    expansion_dict[tokens[i]] = word_expan
        expan = list(set(expan))
        expansion_set = expansion_set | set(expan)
        expansion_lists.append(expan)
    return expansion_lists, expansion_set

# synonym expansion for cnn
# generate a set of synonym to be appended behind each tweet
# the synonym will be vectorized into a separate vector without using ngram
def synonym_expansion2(tagging_results, vocab):
    # target word pos
    print('doing synonym expansion')
    target_tags = set(['N', '^', 'A', 'V'])
    expansion_lists = []
    expansion_dict = {}
    for tagsets in tagging_results:
        flag= True
        try:
            tokens = [t[0] for t in tagsets]
            tags = [t[1] for t in tagsets]
        except:
            flag = False
            print(tagsets)
        expan = []
        if flag:
            for i in range(len(tokens)):
                if tags[i] in target_tags:
                    if tokens[i] in expansion_dict:
                        word_expan = expansion_dict[tokens[i]]
                    else:
                        if tags[i] == 'N' or tags[i] == '^':
                            syn = wordnet.synsets(tokens[i], pos=wordnet.NOUN)
                        elif tags[i] == 'V':
                            syn = wordnet.synsets(tokens[i], pos=wordnet.VERB)
                        elif tags[i] == 'A':
                            syn = wordnet.synsets(tokens[i], pos=wordnet.ADJ)
                        if len(syn) > 0:
                            exp_words = [s.name().rsplit('.', 2)[0] for s in syn]
                            exp_word_count = [(w, vocab[w]) if w in vocab else (w, 1e32) for w in exp_words]
                            if len(exp_word_count) > 3:
                                exp_word_count = sorted(exp_word_count, key=lambda x:x[1])
                                word_expan = [exp_word_count[j][0] for j in range(3)]
                            else:
                                word_expan = [w_c[0] for w_c in exp_word_count]
                    expan += [w for w in word_expan if w not in expan]
                    expansion_dict[tokens[i]] = word_expan
        expansion_lists.append(expan)
    return expansion_dict, expansion_lists

# use abuse indicating terms and drug slang lexicons
# to create features
def term_feature_expansion(token_lists, AITs, DSLs):
    # return sparse matrix ready to be used
    print('doing term feature expansion')
    term_feature_vector = []
    for token_list in token_lists:
        new_vector = []
        # check abuse indicating terms
        ait_count = 0
        for t in AITs:
            if t in token_list:
                ait_count += 1
        new_vector.append(1 if ait_count > 0 else 0)
        new_vector.append(ait_count)
        # check drug slang lexicon
        dsl_count = 0
        for l in DSLs:
            if l in token_list:
                dsl_count += 1
        new_vector.append(1 if ait_count > 0 else 0)
        new_vector.append(ait_count)
        term_feature_vector.append(new_vector)
    # each of these feature should be appended to the vector of corresponding token_list (tweet)
    sparse_vector = csr_matrix(term_feature_vector)
    return sparse_vector

# use word cluster data to create yet another feature
def word_cluster_feature_expansion3(token_lists, word_clusters):
    # use only terms in tweets
    # return csr_matrix ready to use by classifier
    print('doing word cluster feature expansion')
    vector_size = 150 # hard-coded base on word cluster size
    word_clu_features_list = []
    for token_list in token_lists:
        new_vector = np.zeros(vector_size)
        for word in token_list:
            if word in word_clusters:
                new_vector[word_clusters[word]] = 1.0
        word_clu_features_list.append(new_vector)
    sparse_clu_vectors = csr_matrix(word_clu_features_list)
    return sparse_clu_vectors

# train count vectorizer with tweets
def train_vectorizer(tweets, max_features, path, general_params):
    path = general_params['temp_home'] + general_params['tweets_countvectorizer_filename']
    if os.path.isfile(path):
        print('loading tweets vectorizer')
        with open(path, 'rb') as inf:
            vectorizer = pickle.load(inf)
            print(len(vectorizer.vocabulary_))
    else:
        print('building tweets vectorizer')
        vectorizer = CountVectorizer(input='content', encoding='utf-8', decode_error='ignore', ngram_range=(1,3),
            max_df=1.0, min_df=1, max_features=max_features)
        vectorizer.fit(tweets)
        print(len(vectorizer.vocabulary_))
    return vectorizer

# generate (vectorized) features of each tweet in dataset
# for svm, rf models to use
def feature_extraction_workflow_ml():
    # save all results to file for furture access
    # get params for path
    general_params = get_general_params()
    # read word clusters
    word_clusters = read_word_clusters()
    # get tweets (both 5k and filtered 3m)
    labeled_raw_tweets, labeled_clean_tweets, labels = read_labeled_data(general_params, shuffle=False)
    # setup token and tagging filename
    labeled_token_file = general_params['temp_home'] + general_params['5k_hl_token_filename']
    labeled_tagging_file = general_params['temp_home'] + general_params['5k_hl_tagging_filename']
    # clean, tokenize, stem, and tag all tweets
    labeled_token_lists, labeled_tagging_results = clean_tokenize_stem_tweets_baseline(labeled_raw_tweets, labeled_token_file, labeled_tagging_file)
    # reconstruct clean tweets using tokens
    recon_labeled_tweets = [' '.join(tl) for tl in labeled_token_lists]
    # use all tweets to train count vectorizer
    tweets_vectorizer = train_vectorizer(recon_labeled_tweets,
                                                5000,
                  general_params['temp_home'] + general_params['tweets_countvectorizer_filename'],
                  general_params)
    tweet_ngram_vocab = tweets_vectorizer.vocabulary_
    
    # use Counter to build a vocab for synonym expansion
    vocab, inv_vocab, word_frequency_list = build_vocab(labeled_token_lists)
    
    # print(len(inv_vocab))
    # for i in range(len(word_frequency_list)):
    #     if i % 2000 == 0:
    #         print(word_frequency_list[i])
    
    # expand with synonym and train another vectorizer
    labeled_syn_expan_lists, labeled_expansion_set = synonym_expansion(labeled_tagging_results, vocab)

    # reconstruct strings for training count vectorizer with syn expansion results
    recon_labeled_expan_string = [' '.join(sl) for sl in labeled_syn_expan_lists]
    syn_expan_vectorizer = train_vectorizer(recon_labeled_expan_string, 
                                                      2500,
                    general_params['temp_home'] + general_params['syn_expan_countvectorizer_filename'],
                    general_params)
    syn_expan_vocab = syn_expan_vectorizer.vocabulary_
    
    # expand with drug related terms
    labeled_term_feature_vectors = term_feature_expansion(labeled_token_lists, abuse_indicating_terms, drug_slang_lexicon)
    
    # expand with word clusters
    labeled_clu_features_vectors = word_cluster_feature_expansion3(labeled_token_lists, word_clusters)
    
    # put vector and expansions together for classification
    labeled_vectors = tweets_vectorizer.transform(recon_labeled_tweets)
    print(labeled_vectors.shape)

    labeled_syn_expan_vectors = syn_expan_vectorizer.transform(recon_labeled_expan_string)
    print(labeled_syn_expan_vectors.shape)

    print(labeled_term_feature_vectors.shape)

    print(labeled_clu_features_vectors.shape)

    final_labeled_vectors = hstack((labeled_vectors, labeled_syn_expan_vectors, labeled_term_feature_vectors, labeled_clu_features_vectors))
    print(final_labeled_vectors.shape)

    final_hl_vectors_filename = general_params['temp_home'] + general_params['final_labeled_vectors_filename']
    with open(final_hl_vectors_filename, 'wb') as outf:
        pickle.dump(final_labeled_vectors, outf)
    print('feature expansion for ml done')
    # labeled_clu_features_list # cluster expansion (fixed length )
    # labeled_feature_dict # other features
    # #labeled_expansion_lists # syn expan
    # tweets_vectorizer
    # syn_expan_vectorizer
    # [tweets_vector (5000)] + [syn_expan_vector (2500)] + [drug_terms_features (2+2)] + [word_cluster_features (150)]


def feature_extraction_workflow_cnn():
    # go through all tweets since this will take time
    # may need to use multi-processing 
    # save all results to file for furture access
    # get params for path
    general_params = get_general_params()
    # read word clusters
    word_clusters = read_word_clusters()
    # get tweets
    labeled_raw_tweets, labeled_clean_tweets, labels = read_labeled_data(general_params, shuffle=False)
    
    # setup token and tagging filename
    labeled_token_file = general_params['temp_home'] + general_params['5k_hl_token_filename']
    labeled_tagging_file = general_params['temp_home'] + general_params['5k_hl_tagging_filename']
    # clean, tokenize, stem, and tag all tweets
    labeled_token_lists, labeled_tagging_results = clean_tokenize_stem_tweets_baseline(labeled_raw_tweets, labeled_token_file, labeled_tagging_file)

    # reconstruct clean tweets using tokens
    recon_labeled_tweets = [' '.join(tl) for tl in labeled_token_lists]
    
    # use Counter to build a vocab for synonym expansion
    vocab, inv_vocab, word_frequency_list = build_vocab(labeled_token_lists)
    
    # print(len(inv_vocab))
    # for i in range(len(word_frequency_list)):
    #     if i % 2000 == 0:
    #         print(word_frequency_list[i])
    
    # expand with synonym and train another vectorizer
    labeled_syn_expan_dict, labeled_syn_expan_lists = synonym_expansion2(labeled_tagging_results, vocab)
    
    # expand with drug related terms
    labeled_term_feature_vectors = term_feature_expansion(labeled_token_lists, abuse_indicating_terms, drug_slang_lexicon)
    
    # expand with word clusters
    labeled_clu_features_vectors = word_cluster_feature_expansion3(labeled_token_lists, word_clusters)

    print(len(labeled_syn_expan_lists))
    print('saving syn expansion lists')
    with open(general_params['temp_home'] + general_params['hl_syn_expan_lists_filename'], 'wb') as outf:
        pickle.dump(labeled_syn_expan_lists, outf)

    print(len(labeled_syn_expan_dict))
    print('saving syn expansion dict')
    with open(general_params['temp_home'] + general_params['hl_syn_expan_dict_filename'], 'wb') as outf:
        pickle.dump(labeled_syn_expan_dict, outf)

    print(labeled_term_feature_vectors.shape)
    print('saving term feature vectors')
    with open(general_params['temp_home'] + general_params['hl_term_feature_filename'], 'wb') as outf:
        pickle.dump(labeled_term_feature_vectors, outf)

    print(labeled_clu_features_vectors.shape)
    print('saving cluster feature vectors')
    with open(general_params['temp_home'] + general_params['hl_cluster_feature_filename'], 'wb') as outf:
        pickle.dump(labeled_clu_features_vectors, outf)

    print('feature expansion for cnn done')
    # labeled_clu_features_list # cluster expansion (fixed length )
    # labeled_feature_dict # other features
    # #labeled_expansion_lists # syn expan
    # tweets_vectorizer
    # syn_expan_vectorizer
    # [tweets_vector (5000)] + [syn_expan_vector (2500)] + [drug_terms_features (2+2)] + [word_cluster_features (150)]


if __name__ == '__main__':
    feature_extraction_workflow_ml()
    feature_extraction_workflow_cnn()