import unittest
import numpy as np
from config import Config
from team_assignment.team_assigner import TeamAssigner
from transformation.transformer import Transformer
from visualization.renderer import Renderer


class RegressionTests(unittest.TestCase):
    def test_sparse_players_do_not_crash_clustering(self):
        frames = [np.zeros((100, 100, 3), dtype=np.uint8)]
        players = [{1: {'bbox': [-10, -10, 30, 60], 'class_name': 'player'}}]
        TeamAssigner().assigncolors(frames, players)
        self.assertIn(players[0][1]['team'], (0, 1))

    def test_homography_projection_and_missing_keypoint_fallback(self):
        transformer = Transformer.__new__(Transformer)
        transformer.config = Config()
        transformer.smoothing_factor = 0.15
        transformer.last_valid_homography = None
        xy = np.zeros((32, 2), dtype=np.float32)
        conf = np.zeros(32)
        for index in (0, 5, 24, 29):
            xy[index] = transformer.config.PITCH_KEYPOINT_MAP[index]
            conf[index] = 1
        H = transformer.compute_homography({'xy': xy, 'conf': conf})
        np.testing.assert_allclose(transformer.transform_point(52.5, 34, H), [52.5, 34], atol=1e-3)
        self.assertIs(transformer.compute_homography({'xy': None}), H)

    def test_renderer_uses_frame_team_and_shared_homography(self):
        class NoRecompute:
            def compute_homography(self, _):
                raise AssertionError('Homography was recomputed')
        frame = np.zeros((240, 400, 3), dtype=np.uint8)
        tracks = {'players': [{1: {'bbox': [10, 20, 30, 60], 'team': 1}}],
                  'referees': [{}], 'ball': [{}]}
        assigner = TeamAssigner()
        assigner.player_team_dict[1] = 0
        result = Renderer(Config()).draw([frame], tracks, assigner, NoRecompute(),
                                         [{'xy': None}], homographies=[None])
        self.assertTrue(np.any(np.all(result[0] == Config().TEAM_1_COLOR, axis=2)))
        self.assertFalse(frame.any())

if __name__ == '__main__':
    unittest.main()
