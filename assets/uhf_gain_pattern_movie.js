  const movie = document.createElement('canvas');
  movie.width = 1280;
  movie.height = 720;
  document.body.append(movie);
  const mc = movie.getContext('2d',{alpha:false});
  const left = 48, rightEdge = 1232, paneWidth = 384;
  const title = {fairbanks:'Fairbanks pass', boulder:'Boulder pass'};
  const paneTitles = ['Total gain','RHCP gain','LHCP gain'];
  const color = name => rgb(palette[name]);
  const muted = () => rgb(mix(palette.background,palette.foreground,.65));

  function write(text,x,y,size=21,{align='left',ink=color('foreground'),weight=400}={}) {
    mc.font = weight+' '+size+'px -apple-system, BlinkMacSystemFont, Arial, sans-serif';
    mc.textAlign = align;
    mc.textBaseline = 'middle';
    mc.fillStyle = ink;
    mc.fillText(text,x,y);
  }
  function fillRect(x,y,w,h,ink) {
    mc.fillStyle = ink;
    mc.fillRect(x,y,w,h);
  }
  function directionAt(rows,seconds) {
    let lo=0, hi=rows.length-1;
    while (hi-lo>1) {
      const i=(hi+lo)>>1;
      if (rows[i][0]<=seconds) lo=i;
      else hi=i;
    }
    const a=rows[lo], b=rows[hi];
    const q=clamp((seconds-a[0])/(b[0]-a[0]),0,1);
    return unit(mix(a.slice(1,4),b.slice(1,4),q));
  }
  window.movieFrame = function(scene,seconds,yawDegrees,pitchDegrees) {
    const track = tracks[scene];
    const stop = track.samples.at(-1)[0];
    const vector = directionAt(track.samples,seconds);
    selection = {vector,gain:gainAt(vector)};
    yaw=yawDegrees*deg;
    pitch=pitchDegrees*deg;
    const camera = makeCamera();

    fillRect(0,0,movie.width,movie.height,color('background'));
    write('SunCET UHF antenna pattern',left,51,32,{weight:500});
    write(title[scene],left,91,24,{weight:500});
    write('Pass elapsed  '+clock(seconds)+' / '+clock(stop),rightEdge,91,22,{align:'right',ink:muted()});
    const share = selection.gain[1]/selection.gain[0];
    write('RHCP '+percent(share),left,134,21,{ink:color('rh')});
    write('LHCP '+percent(1-share),rightEdge,134,21,{align:'right',ink:color('lh')});
    const barX=left+158, barW=rightEdge-left-316;
    fillRect(barX,130,barW,8,color('lh'));
    fillRect(barX,130,barW*share,8,color('rh'));

    canvases.forEach((canvas,mode) => {
      const x=left+mode*(paneWidth+16);
      write(paneTitles[mode],x,183,22,{weight:500});
      write(signed(db(selection.gain[mode])),x+paneWidth,183,23,{align:'right'});
      drawSurface(mode,camera,false);
      drawOverlay(mode,true);
      mc.drawImage(canvas,x,198,paneWidth,375);
    });
    const fg=color('foreground'), bg=color('background');
    arrow(mc,[left,597],[left+30,597],fg,bg);
    write('Spacecraft-to-station direction',left+42,597,19,{ink:muted()});
    mc.beginPath();
    mc.arc(left+359,597,5,0,2*Math.PI);
    mc.strokeStyle=fg;
    mc.lineWidth=1.8;
    mc.stroke();
    write('Gain along arrow',left+372,597,19,{ink:muted()});
    mc.setLineDash([4,4]);
    line(mc,[left+570,597],[left+599,597],muted(),1.3);
    mc.setLineDash([]);
    write('0 dBi reference',left+611,597,19,{ink:muted()});
    write('Shared scale · −20 dBi floor',rightEdge,597,19,{align:'right',ink:muted()});

    const nearNull = scene==='fairbanks' && Math.abs(seconds-track.worst)<9;
    const rolling = nearest(track.samples,seconds)[4];
    const status = nearNull ? 'Near the RHCP minimum · no spacecraft roll' :
      rolling ? 'Spacecraft roll in progress' : 'Spacecraft body frame';
    write(status,left,643,20,{ink:nearNull||rolling?fg:muted()});
    const timelineWidth=rightEdge-left, timelineY=670;
    fillRect(left,timelineY,timelineWidth,4,rgb(mix(palette.background,palette.foreground,.2)));
    fillRect(left,timelineY,timelineWidth*seconds/stop,4,muted());
    if (scene==='fairbanks') {
      const mark=left+timelineWidth*track.worst/stop;
      fillRect(mark-1,timelineY-4,2,12,color('lh'));
    } else if (track.roll) {
      const [first,last]=track.roll;
      fillRect(left+timelineWidth*first/stop,timelineY,timelineWidth*(last-first)/stop,4,color('rh'));
    }
    const current=left+timelineWidth*seconds/stop;
    mc.beginPath();
    mc.arc(current,timelineY+2,5,0,2*Math.PI);
    mc.fillStyle=fg;
    mc.fill();
    return movie.toDataURL('image/jpeg',.95).split(',')[1];
  };
