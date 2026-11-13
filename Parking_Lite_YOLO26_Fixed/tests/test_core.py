import tempfile, unittest
from pathlib import Path
import numpy as np
from parking_lite.core import save_layout,load_layout,analyze_image,draw_analysis
class TestParking(unittest.TestCase):
    def setUp(self): self.im=np.zeros((400,600,3),dtype=np.uint8)
    def test_mapped(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'layout.json'
            save_layout(p,[[(50,50),(250,50),(250,350),(50,350)],[(300,50),(500,50),(500,350),(300,350)]],600,400)
            layout=load_layout(p)
            vehicles=[{'label':'car','confidence':.9,'box':[100,100,200,300]}]
            result=analyze_image(self.im,vehicles,layout)
            self.assertEqual(result['counts'],{'total':2,'occupied':1,'available':0,'unknown':1})
            self.assertEqual(result['spaces'][0]['status'],'occupied')
            self.assertEqual(result['spaces'][1]['status'],'unknown')
            self.assertEqual(draw_analysis(self.im,result).shape,self.im.shape)
    def test_unmapped(self):
        result=analyze_image(self.im,[],None)
        self.assertFalse(result['mapped'])
        self.assertEqual(result['counts']['total'],0)
    def test_invalid_polygon(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError): save_layout(Path(d)/'a.json',[[(1,2),(3,4)]],600,400)
if __name__=='__main__': unittest.main()
