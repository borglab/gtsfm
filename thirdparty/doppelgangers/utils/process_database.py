from pathlib import Path
import numpy as np
import sqlite3
import shutil
from scipy.special import softmax

from .database import pair_id_to_image_ids, blob_to_array
from .database import COLMAPDatabase


def read_two_view_from_db(database_path):
    db = COLMAPDatabase.connect(database_path)
    id2name = db.image_id_to_name()
    pairs = []
    matches = []
    H_list = []
    E_list = []
    F_list = []
    pairs_id = []

    for pair_id, rows, cols, data, config, F, E, H, qvec, tvec in db.execute("SELECT pair_id, rows, cols, data, config, F, E, H, qvec, tvec FROM two_view_geometries"):
        id1, id2 = pair_id_to_image_ids(pair_id)
        name1, name2 = id2name[id1], id2name[id2]
        if data is None:
            continue
        pairs_id.append(pair_id)
        pairs.append((name1, name2))
        match = blob_to_array(data, np.uint32, (rows, cols))
        matches.append(match)
        F = blob_to_array(F, np.float64).reshape(-1,3)
        E = blob_to_array(E, np.float64).reshape(-1,3)
        H = blob_to_array(H, np.float64).reshape(-1,3)
        qvec = blob_to_array(qvec, np.float64)
        tvec = blob_to_array(tvec, np.float64)
        H_list.append(H)
        F_list.append(F)
        E_list.append(E)
    db.close()
    return pairs, matches, E_list, F_list, H_list, pairs_id


def deleteMultipleRecords(database_path, match_pair_list, table="two_view_geometries"):
    match_pair_list = [(i, ) for i in match_pair_list]
    try:
        db = COLMAPDatabase.connect(database_path)
        cursor = db.cursor()
        sqlite_update_query = """DELETE from %s where pair_id = ?""" % table
        cursor.executemany(sqlite_update_query, match_pair_list)
        db.commit()
        print("Total", cursor.rowcount, "Records deleted successfully")
        cursor.close()

    except sqlite3.Error as error:
        print("Failed to delete multiple records from sqlite table", error)
    finally:
        if db:
            db.close()


def create_image_pair_list(db_path, output_path):
    # output image pair list file: 
    # path to image 1, path to image 2, label (for doppelganger dataset), number of matches, pair_id
    pairs_list = []
    pairs, matches, _, _, _, pairs_id = read_two_view_from_db(db_path)
    for i in range(len(pairs_id)):
        name1, name2 = pairs[i]
        label = 0
        pairs_list.append([name1, name2, label, matches[i].shape[0], pairs_id[i]])
    pairs_list = np.concatenate(pairs_list, axis=0).reshape(-1, 5)
    np.save('%s/pairs_list.npy' % output_path, pairs_list)    
    return '%s/pairs_list.npy' % output_path


def remove_doppelgangers(db_path, pair_probability_file, pair_path, threshold):
    # remove doppelgangers pairs from colmap database
    new_db_path = db_path.replace('.db', '_threshold_%.3f.db'%threshold)
    shutil.copyfile(db_path, new_db_path)
    
    result = np.load(pair_probability_file, allow_pickle=True).item()
    y_scores = np.array(result['prob']).reshape(-1,2)
    y_scores = softmax(y_scores, axis=1)[:, 1]
    
    pairs_info = np.load(pair_path)
    pairs_id = np.array(pairs_info)[:, -1]

    print('number of matches in database: ', len(y_scores))

    match_pair_list = []
    for i in range(len(pairs_id)):
        if y_scores[i] < threshold:
                match_pair_list.append(pairs_id[i])   
    deleteMultipleRecords(new_db_path, match_pair_list, table="matches")
    deleteMultipleRecords(new_db_path, match_pair_list, table="two_view_geometries")
    return new_db_path
    
