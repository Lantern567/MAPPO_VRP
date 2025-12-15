import { useState, useEffect, useRef } from 'react'
import './App.css'

function App() {
  const [data, setData] = useState(null)
  const [currentStep, setCurrentStep] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [playSpeed, setPlaySpeed] = useState(200)
  const canvasRef = useRef(null)

  // Load episode data
  useEffect(() => {
    fetch('/episode_data.json')
      .then(res => res.json())
      .then(d => {
        setData(d)
        setCurrentStep(0)
      })
      .catch(err => console.error('Failed to load data:', err))
  }, [])

  // Auto-play
  useEffect(() => {
    if (!isPlaying || !data) return
    const interval = setInterval(() => {
      setCurrentStep(prev => {
        if (prev >= data.timesteps.length - 1) {
          setIsPlaying(false)
          return prev
        }
        return prev + 1
      })
    }, playSpeed)
    return () => clearInterval(interval)
  }, [isPlaying, data, playSpeed])

  // Draw canvas
  useEffect(() => {
    if (!data || !canvasRef.current) return
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    const width = canvas.width
    const height = canvas.height

    // Clear
    ctx.fillStyle = '#1a1a2e'
    ctx.fillRect(0, 0, width, height)

    // Convert coordinates (-1, 1) to canvas
    const toCanvasX = x => (x + 1) * width / 2
    const toCanvasY = y => (1 - y) * height / 2  // Flip Y

    const timestep = data.timesteps[currentStep]

    // Draw route nodes (gray circles)
    ctx.fillStyle = '#444'
    data.route_nodes.forEach(node => {
      ctx.beginPath()
      ctx.arc(toCanvasX(node.x), toCanvasY(node.y), 8, 0, Math.PI * 2)
      ctx.fill()
    })

    // Draw customers
    data.customers.forEach((customer, i) => {
      const served = timestep.customers_served.includes(i)
      ctx.fillStyle = served ? '#4ade80' : '#ef4444'
      ctx.beginPath()
      ctx.arc(toCanvasX(customer.x), toCanvasY(customer.y), 12, 0, Math.PI * 2)
      ctx.fill()
      // Label
      ctx.fillStyle = '#fff'
      ctx.font = '10px Arial'
      ctx.textAlign = 'center'
      ctx.fillText(`C${i}`, toCanvasX(customer.x), toCanvasY(customer.y) + 4)
    })

    // Draw truck (blue square)
    const truck = timestep.truck
    ctx.fillStyle = '#3b82f6'
    ctx.fillRect(toCanvasX(truck.x) - 15, toCanvasY(truck.y) - 10, 30, 20)
    ctx.fillStyle = '#fff'
    ctx.font = '10px Arial'
    ctx.textAlign = 'center'
    ctx.fillText('Truck', toCanvasX(truck.x), toCanvasY(truck.y) + 4)

    // Draw drones
    const droneColors = ['#f59e0b', '#8b5cf6', '#ec4899']
    timestep.drones.forEach((drone, i) => {
      ctx.fillStyle = droneColors[i % droneColors.length]
      ctx.beginPath()
      ctx.moveTo(toCanvasX(drone.x), toCanvasY(drone.y) - 10)
      ctx.lineTo(toCanvasX(drone.x) - 8, toCanvasY(drone.y) + 8)
      ctx.lineTo(toCanvasX(drone.x) + 8, toCanvasY(drone.y) + 8)
      ctx.closePath()
      ctx.fill()
      // Battery indicator
      ctx.fillStyle = drone.battery > 0.3 ? '#4ade80' : '#ef4444'
      ctx.fillRect(toCanvasX(drone.x) - 15, toCanvasY(drone.y) + 12, 30 * drone.battery, 4)
      ctx.strokeStyle = '#888'
      ctx.strokeRect(toCanvasX(drone.x) - 15, toCanvasY(drone.y) + 12, 30, 4)
    })

    // Draw legend
    ctx.fillStyle = '#fff'
    ctx.font = '12px Arial'
    ctx.textAlign = 'left'
    ctx.fillText(`Step: ${currentStep}/${data.timesteps.length - 1}`, 10, 20)
    ctx.fillText(`Reward: ${timestep.reward.toFixed(2)}`, 10, 40)

  }, [data, currentStep])

  if (!data) {
    return (
      <div className="app">
        <h1>VRP MAPPO Visualization</h1>
        <p>Loading episode data...</p>
        <p style={{color: '#888', fontSize: '14px'}}>
          Run <code>python mappo/scripts/run_demo.py</code> to generate data
        </p>
      </div>
    )
  }

  return (
    <div className="app">
      <h1>VRP MAPPO Visualization</h1>

      <div className="canvas-container">
        <canvas ref={canvasRef} width={600} height={600} />
      </div>

      <div className="controls">
        <button onClick={() => setCurrentStep(Math.max(0, currentStep - 1))}>
          ⏮ Prev
        </button>
        <button onClick={() => setIsPlaying(!isPlaying)}>
          {isPlaying ? '⏸ Pause' : '▶ Play'}
        </button>
        <button onClick={() => setCurrentStep(Math.min(data.timesteps.length - 1, currentStep + 1))}>
          Next ⏭
        </button>
        <button onClick={() => setCurrentStep(0)}>
          ⏹ Reset
        </button>
      </div>

      <div className="slider-container">
        <label>Step: {currentStep}</label>
        <input
          type="range"
          min={0}
          max={data.timesteps.length - 1}
          value={currentStep}
          onChange={e => setCurrentStep(Number(e.target.value))}
        />
      </div>

      <div className="slider-container">
        <label>Speed: {playSpeed}ms</label>
        <input
          type="range"
          min={50}
          max={500}
          value={playSpeed}
          onChange={e => setPlaySpeed(Number(e.target.value))}
        />
      </div>

      <div className="info-panel">
        <h3>Episode Info</h3>
        <p>Drones: {data.config.num_drones} | Customers: {data.config.num_customers}</p>
        <p>Total Steps: {data.summary.total_steps} | Total Reward: {data.summary.total_reward.toFixed(2)}</p>
        <p>Served: {data.summary.customers_served}/{data.summary.total_customers}</p>

        <h3>Current Step Details</h3>
        <div className="details">
          <p><strong>Truck:</strong> ({data.timesteps[currentStep].truck.x.toFixed(2)}, {data.timesteps[currentStep].truck.y.toFixed(2)})</p>
          {data.timesteps[currentStep].drones.map((d, i) => (
            <p key={i}>
              <strong>Drone {i}:</strong> ({d.x.toFixed(2)}, {d.y.toFixed(2)}) | Battery: {(d.battery * 100).toFixed(0)}% | {d.status}
            </p>
          ))}
        </div>
      </div>

      <div className="legend">
        <span><span className="dot blue"></span> Truck</span>
        <span><span className="dot orange"></span> Drone</span>
        <span><span className="dot red"></span> Customer (waiting)</span>
        <span><span className="dot green"></span> Customer (served)</span>
        <span><span className="dot gray"></span> Route Node</span>
      </div>
    </div>
  )
}

export default App
