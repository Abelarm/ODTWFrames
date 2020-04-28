
specs = {
    'gunpoint': {
        'y_dim': 2,
        'x_dim': 150,
        'channels': 2,
        'ref_id': [0, 6],
        'repre_samples_25': ['X:0_840-865|Y:1.npy', 'X:0_685-710|Y:2.npy'],
        'repre_samples_15': ['X:0_845-860|Y:1.npy', 'X:0_690-705|Y:2.npy'],
        'repre_samples_5':  ['X:0_847-852|Y:1.npy', 'X:0_697-702|Y:2.npy'],
    },
    'gunpoint_base': {
        'y_dim': 2,
        'x_dim': 150,
        'channels': 7,
        'ref_id': [0, 1, 2, 3, 4, 5, 6],
        'max_stream_id': [20, 5, 10],
        'repre_samples_25': ['X:0_840-865|Y:1.npy', 'X:0_685-710|Y:2.npy'],
        'repre_samples_15': ['X:0_845-860|Y:1.npy', 'X:0_690-705|Y:2.npy'],
        'repre_samples_5':  ['X:0_847-852|Y:1.npy', 'X:0_697-702|Y:2.npy'],
    },
    'cbf': {
        'y_dim': 3,
        'x_dim': 100,
        'channels': 3,
        'ref_id': [8, 13, 25],
        'repre_samples_25': ['1/0_1130-1155.png', '2/0_330-355.png', '3/0_1430-1455.png'],
        'repre_samples_15': ['1/0_1130-1145.png', '2/0_330-345.png', '3/0_1430-1445.png'],
        'repre_samples_5':  ['1/0_1128-1133.png', '2/0_349-354.png', '3/0_1433-1438.png']
    },
    'cbf_base': {
        'y_dim': 3,
        'x_dim': 100,
        'channels': 7,
        'ref_id': [0, 1, 2, 3, 4, 5, 6],
        'max_stream_id': [20, 5, 10],
        'repre_samples_25': ['X:0_1130-1155|Y:1.npy', 'X:0_330-355|Y:2.npy', 'X:0_1430-1455|Y:3.npy'],
        'repre_samples_15': ['X:0_1130-1145|Y:1.npy', 'X:0_330-345|Y:2.npy', 'X:0_1430-1445|Y:3.npy'],
        'repre_samples_5':  ['X:0_1128-1133|Y:1.npy', 'X:0_349-355|Y:2.npy', 'X:0_1433-1438|Y:3.npy']
    },
    'rational': {
        'y_dim': 4,
        'x_dim': 100,
        'channels': 4,
        'ref_id': [6, 17, 21, 34],
        'repre_samples_25': ['1/0_560-585.png', '2/0_1130-1155.png', '3/0_640-665.png', '4/0_240-265.png'],
        'repre_samples_15': ['1/0_560-575.png', '2/0_1130-1145.png', '3/0_640-655.png', '4/0_240-255.png'],
        'repre_samples_5':  ['1/0_562-567.png', '2/0_1133-1138.png', '3/0_641-646.png', '4/0_241-246.png'],
    },
    'rational_base': {
        'y_dim': 4,
        'x_dim': 100,
        'channels': 7,
        'ref_id': [0, 1, 2, 3, 4, 5, 6],
        'max_stream_id': [20, 5, 10],
        'repre_samples_5':  ['X:0_562-567|Y:1.npy', 'X:0_1133-1138|Y:2.npy', 'X:0_641-646|Y:3.npy',
                             'X:0_241-246|Y:4.npy']
    },
    'synthetic_control': {
        'y_dim': 6,
        'x_dim': 100,
        'channels': 6,
    }
}
