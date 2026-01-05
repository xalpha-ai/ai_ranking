# Railway Deployment Guide

This guide will help you deploy the AI Visibility Score System to Railway with PostgreSQL database.

## Prerequisites

- GitHub account
- Railway account (sign up at https://railway.app)
- Your code pushed to a GitHub repository

## Step 1: Prepare Your Repository

1. Make sure all files are committed to your GitHub repository:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
   git push -u origin main
   ```

## Step 2: Create Railway Project

1. Go to https://railway.app
2. Click "Start a New Project"
3. Select "Deploy from GitHub repo"
4. Authorize Railway to access your GitHub account
5. Select your repository

## Step 3: Add PostgreSQL Database

1. In your Railway project, click "+ New"
2. Select "Database" → "PostgreSQL"
3. Railway will automatically:
   - Create a PostgreSQL database
   - Add `DATABASE_URL` environment variable
   - Link it to your web service

## Step 4: Configure Environment Variables

Railway automatically provides `DATABASE_URL`, but you can add others if needed:

1. Click on your web service
2. Go to "Variables" tab
3. Add any custom variables:
   - `FLASK_ENV=production` (optional)
   - `SECRET_KEY=your-secret-key` (optional)

**Note:** API keys are provided by users through the UI, not stored in environment variables.

## Step 5: Deploy

1. Railway will automatically detect your `Procfile` and deploy
2. The build process will:
   - Install dependencies from `requirements.txt`
   - Run database migrations automatically
   - Start the web server using Gunicorn

## Step 6: Access Your Application

1. Once deployed, Railway will provide a URL like: `https://your-app.up.railway.app`
2. You can also add a custom domain:
   - Go to "Settings" tab
   - Click "Generate Domain" or "Add Custom Domain"
   - For xalpha-ai.com:
     - Add `dashboard.xalpha-ai.com` or subdomain of your choice
     - Update DNS records as instructed by Railway

## File Structure Explanation

- **Procfile**: Tells Railway to run `gunicorn app:app`
- **requirements.txt**: Lists all Python dependencies
- **runtime.txt**: Specifies Python version (3.9.13)
- **database.py**: Database models and connection logic
- **.env.example**: Template for environment variables (not deployed)
- **.gitignore**: Prevents committing sensitive files

## Database Initialization

The database tables are created automatically on first run:
- `brands` - Master brand table
- `visibility_scores` - Historical visibility scores
- `competitive_analysis` - Historical competitive analysis
- `ranking_analysis` - Historical ranking analysis
- `geographic_scores` - Historical geographic presence

## Monitoring

1. View logs:
   - Go to your service in Railway
   - Click "Deployments" tab
   - Click on latest deployment
   - View real-time logs

2. Check database:
   - Click on PostgreSQL service
   - Go to "Data" tab
   - Query your tables

## Updating Your App

When you push changes to GitHub:
```bash
git add .
git commit -m "Your update message"
git push
```

Railway will automatically:
1. Detect the push
2. Build the new version
3. Deploy with zero downtime

## Troubleshooting

### Database Connection Issues
- Check that `DATABASE_URL` variable exists
- Railway automatically converts `postgres://` to `postgresql://`
- Check logs for database connection errors

### Import Errors
- Make sure all dependencies are in `requirements.txt`
- Check Python version matches `runtime.txt`

### Application Not Starting
- Check logs for error messages
- Verify `Procfile` syntax
- Ensure `app.py` exists and has `app` variable

## Cost Estimate

Railway pricing (as of 2024):
- **Free Trial**: $5 credit/month (enough for demo)
- **PostgreSQL**: ~$5/month after free trial
- **Web Service**: ~$5/month
- **Total**: ~$10/month for production

For demo: Use the free $5 credit (no credit card required)

## Security Notes

1. Never commit `.env` files (already in `.gitignore`)
2. API keys are user-provided, not stored server-side
3. Use Railway's secret management for sensitive data
4. Enable HTTPS (automatic with Railway domains)

## Next Steps

1. Test all features after deployment
2. Set up custom domain (dashboard.xalpha-ai.com)
3. Monitor usage and costs
4. Set up error monitoring (optional: Sentry integration)

## Support

- Railway Docs: https://docs.railway.app
- Railway Discord: https://discord.gg/railway
- Project Issues: GitHub Issues

---

**Your app is now live and ready to track AI visibility scores!**
