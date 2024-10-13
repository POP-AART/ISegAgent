import json

""" This script is used to calculate the nsd or dice  for a specific point."""
def cal_score_at_point(fix_point_idx='1'):
    
    nsd_file_path = './union_out_nsd.json'
    dice_file_path = './union_out_dice.json'
    
    with open(dice_file_path, 'r') as file:
        dice_data = json.load(file)['tumor']
        
    with open(nsd_file_path, 'r') as file:
        nsd_data = json.load(file)['tumor']  

    dice_list = []
    for one_data, score in dice_data.items():
        for point_idx, point_dice in score.items():
            if point_idx == fix_point_idx:
                dice_list.append(point_dice)
    
    nsd_list = []
    for one_data, score in nsd_data.items():
        for point_idx, point_nsd in score.items():
            if point_idx == fix_point_idx:
                nsd_list.append(point_nsd)

    mean_dice = sum(dice_list) / len(dice_list)
    mean_nsd = sum(nsd_list) / len(nsd_list)
    
    print('point: {}   dice: {:.4f}   nsd: {:.4f}'.format(int(fix_point_idx) + 1, mean_dice, mean_nsd))


if __name__ == '__main__':
    
    for i in range(5):
        cal_score_at_point(str(i))

