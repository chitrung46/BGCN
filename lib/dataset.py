import pandas as pd

def load_dataset(args):
    if args.dataset == 'All_Beauty':
        df = pd.read_json('./data/All_Beauty.jsonl.gz', compression='gzip', lines=True)
        df.drop(columns=['title', 'images', 'asin'], inplace=True)
        df.rename(columns={'parent_asin': 'item_id'}, inplace=True)
        df = df[df['verified_purchase'] == True]

    Seq = dict()
    user_map = dict()
    item_map = dict()
    user_num = 0
    item_num = 0

    for index, row in df.iterrows():
        user_id = row['user_id']
        item_id = row['item_id']
        rating = row['rating']
        review = row['text']
        timestamp = row['timestamp']

        # mapping
        if user_id not in user_map:
            user_map[user_id] = user_num
            user_num += 1
        if item_id not in item_map:
            item_map[item_id] = item_num
            item_num += 1

        uidx = user_map[user_id]
        iidx = item_map[item_id]

        if uidx in Seq.keys():
            Seq[uidx].append([iidx, rating, review, timestamp])
        else:
            Seq[uidx] = [[iidx, rating, review, timestamp]]

    for seq in Seq.values():
        seq.sort(key=lambda x: x[3])        

    print(f"\nNumber of users before filtering: {len(Seq)}")

    min_seq_length = args.min_seq_length
    max_seq_length = args.max_seq_length

    filtered_Seq = {}
    for user_id, seq in Seq.items():
        if len(seq) >= min_seq_length:
            # Take the last max_seq_length items
            processed_seq = seq[-max_seq_length:]

            if len(processed_seq) < max_seq_length:
                padding_length = max_seq_length - len(processed_seq)
                padding = [(0, '', pd.Timestamp(0))] * padding_length
                processed_seq = padding + processed_seq

            filtered_Seq[user_id] = processed_seq

    print(f"\nNumber of users after filtering: {len(filtered_Seq)}")
    return filtered_Seq

            