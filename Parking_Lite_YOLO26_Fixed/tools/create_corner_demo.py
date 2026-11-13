"""Generate a controlled geometric test pattern, not a real parking photograph."""
from pathlib import Path
import cv2
import numpy as np


def main():
    root=Path(__file__).resolve().parents[1]
    image=np.full((700,1100,3),55,np.uint8)
    cv2.putText(image,'SYNTHETIC CORNER-MARKING TEST - NO CARS',(60,65),
                cv2.FONT_HERSHEY_SIMPLEX,.8,(100,100,100),2,cv2.LINE_AA)
    for x in (110,380,650):
        for a,b,sx,sy in [(x,140,1,1),(x+160,140,-1,1),(x+160,470,-1,-1),(x,470,1,-1)]:
            cv2.line(image,(a,b),(a+sx*25,b),(245,245,245),5)
            cv2.line(image,(a,b),(a,b+sy*25),(245,245,245),5)
    output=root/'samples'/'parking photos'/'corner-brackets-demo.png'
    if output.exists():raise SystemExit('Demo already exists; not overwritten')
    output.write_bytes(cv2.imencode('.png',image)[1].tobytes())
    print(output)


if __name__=='__main__':main()
