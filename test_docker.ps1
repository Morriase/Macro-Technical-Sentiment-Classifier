# Quick Docker validation for Windows
# Tests Docker build locally before pushing to Render

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "QUICK DOCKER VALIDATION" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "This will:"
Write-Host "  1. Build Docker image locally"
Write-Host "  2. Test that it starts"
Write-Host "  3. Test health endpoint"
Write-Host ""
Write-Host "Time: ~3 minutes (vs 15+ on Render)" -ForegroundColor Yellow
Write-Host ""

# Check if Docker is running
try {
    docker info | Out-Null
    Write-Host "✓ Docker is running" -ForegroundColor Green
}
catch {
    Write-Host "❌ Docker is not running!" -ForegroundColor Red
    Write-Host "   Start Docker Desktop and try again."
    exit 1
}

Write-Host ""

# Build image
Write-Host "Step 1: Building Docker image..." -ForegroundColor Cyan
Write-Host "---------------------------------------"
docker build -t forex-test:latest .
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "❌ Docker build failed!" -ForegroundColor Red
    Write-Host "   Fix the errors above before pushing to Render."
    exit 1
}

Write-Host ""
Write-Host "✓ Docker build successful" -ForegroundColor Green
Write-Host ""

# Test run
Write-Host "Step 2: Starting container..." -ForegroundColor Cyan
Write-Host "---------------------------------------"
$CONTAINER_ID = docker run -d -p 5555:10000 -e PORT=10000 forex-test:latest

if ([string]::IsNullOrEmpty($CONTAINER_ID)) {
    Write-Host "❌ Failed to start container" -ForegroundColor Red
    exit 1
}

Write-Host "✓ Container started: $CONTAINER_ID" -ForegroundColor Green
Write-Host ""

# Wait for startup
Write-Host "Step 3: Waiting for server to start (30 seconds)..." -ForegroundColor Cyan
Write-Host "---------------------------------------"
Start-Sleep -Seconds 30

# Check logs
Write-Host ""
Write-Host "Container logs:" -ForegroundColor Cyan
Write-Host "---------------------------------------"
docker logs $CONTAINER_ID 2>&1 | Select-Object -Last 20
Write-Host "---------------------------------------"
Write-Host ""

# Test health endpoint
Write-Host "Step 4: Testing health endpoint..." -ForegroundColor Cyan
Write-Host "---------------------------------------"
try {
    $response = Invoke-WebRequest -Uri "http://localhost:5555/health" -UseBasicParsing -TimeoutSec 10
    $json = $response.Content | ConvertFrom-Json
    
    Write-Host "Response:" -ForegroundColor Green
    $json | ConvertTo-Json -Depth 10
    Write-Host ""
    
}
catch {
    Write-Host "❌ No response from health endpoint" -ForegroundColor Red
    Write-Host ""
    Write-Host "Full logs:"
    docker logs $CONTAINER_ID
    docker stop $CONTAINER_ID | Out-Null
    docker rm $CONTAINER_ID | Out-Null
    exit 1
}

# Cleanup
Write-Host "Step 5: Cleaning up..." -ForegroundColor Cyan
Write-Host "---------------------------------------"
docker stop $CONTAINER_ID | Out-Null
docker rm $CONTAINER_ID | Out-Null
Write-Host "✓ Container stopped and removed" -ForegroundColor Green
Write-Host ""

# Success!
Write-Host "==========================================" -ForegroundColor Green
Write-Host "✅ VALIDATION SUCCESSFUL!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Your Docker image works locally."
Write-Host "It should work on Render too."
Write-Host ""
Write-Host "Next steps:"
Write-Host "  git add ."
Write-Host "  git commit -m 'Your message'"
Write-Host "  git push origin main"
Write-Host ""
